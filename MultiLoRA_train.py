import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.resnet import BasicBlock,Bottleneck
import torchvision
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.models.mobilenetv2 import Conv2dNormActivation
from vit_model import ViTSelfAttention,Mlp
from transformers.models.llama.modeling_llama import LlamaDecoderLayer,LlamaAttention,LlamaMLP
from transformers.models.opt.modeling_opt import OPTDecoderLayer,OPTAttention
from maskmanager import MaskManager
class LoRABase(nn.Module):
    def __init__(self, block_idx=None, layer_name=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_group_indexes = [] 
        # self.pruning_mask = None
        self.block_idx = block_idx  
        self.layer_name = layer_name  
        self.lora_enabled = True
    # def set_pruning_mask(self, mask):
    #     """设置剪枝掩码"""
    #     self.pruning_mask = mask
    def get_pruning_mask(self, device=None):
        """从全局管理器获取剪枝掩码"""
        if self.block_idx is not None and self.layer_name is not None:
            return MaskManager.get_mask(self.block_idx, self.layer_name, device)
        return None
    def set_active_group_ranks(self, target_indexes):
        if target_indexes is None:
            self.active_group_indexes = []
        else:
            self.active_group_indexes = target_indexes
    def get_active_indexes(self):
        """获取当前激活秩对应的索引"""
        # print("当前组使用的秩")
        # print(self.active_group_indexes)
        return self.active_group_indexes
    def set_lora_enabled(self, enabled=True):
        """设置LoRA启用状态"""
        self.lora_enabled = enabled
class _LoRA_qkv_timm(LoRABase):
    def __init__(self, qkv: nn.Module, prune_rate: float, s: int = 1):
        super().__init__()
        self.qkv = qkv
        self.weight = qkv.weight
        self.bias = qkv.bias
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features
        self.prune_rate = prune_rate
        self.dim = qkv.out_features // 3
        self.s = s

        # 动态生成LoRA配置
        self.initial_ranks = [8,8,8]
        self.extended_ranks = []
        self.active_ranks = []
        
        # 使用ParameterList存储LoRA参数
        self.q_lora_down_list = nn.ParameterList()
        self.q_lora_up_list = nn.ParameterList()
        self.v_lora_down_list = nn.ParameterList()
        self.v_lora_up_list = nn.ParameterList()
        
        for r in self.initial_ranks:
            # Query LoRA参数
            q_down = nn.Parameter(torch.zeros(self.dim, r))
            q_up = nn.Parameter(torch.zeros(r, self.dim))
            nn.init.kaiming_uniform_(q_up, a=math.sqrt(5))
            nn.init.zeros_(q_down)
            self.q_lora_down_list.append(q_down)
            self.q_lora_up_list.append(q_up)
            
            # Value LoRA参数
            v_down = nn.Parameter(torch.zeros(self.dim, r))
            v_up = nn.Parameter(torch.zeros(r, self.dim))
            nn.init.kaiming_uniform_(v_up, a=math.sqrt(5))
            nn.init.zeros_(v_down)
            self.v_lora_down_list.append(v_down)
            self.v_lora_up_list.append(v_up)
        
        self.weight.requires_grad = False

    def set_rank_requires_grad(self, target_indexes, requires_grad=True):   
        for idx in range(len(self.initial_ranks)):
            if idx in target_indexes:
                self.q_lora_down_list[idx].requires_grad = requires_grad
                self.q_lora_up_list[idx].requires_grad = requires_grad
                self.v_lora_down_list[idx].requires_grad = requires_grad
                self.v_lora_up_list[idx].requires_grad = requires_grad
            else:
                self.q_lora_down_list[idx].requires_grad = False
                self.q_lora_up_list[idx].requires_grad = False
                self.v_lora_down_list[idx].requires_grad = False
                self.v_lora_up_list[idx].requires_grad = False

    def forward(self, x):
        # 获取原始权重
        weight = self.weight  # (3*dim, dim)
        
        # 将权重拆分为q, k, v三部分
        dim = self.dim
        W_q = weight[:dim, :]
        W_k = weight[dim:2*dim, :]
        W_v = weight[2*dim:, :]
        
        # 初始化ΔW_q和ΔW_v
        delta_W_q = torch.zeros_like(W_q)
        delta_W_v = torch.zeros_like(W_v)
        
        # 应用激活的LoRA组
        for idx in self.get_active_indexes():
            # 计算当前组的LoRA更新
            
            q_lora = self.q_lora_down_list[idx] @ self.q_lora_up_list[idx]
            v_lora = self.v_lora_down_list[idx] @ self.v_lora_up_list[idx]
            
            # 累加LoRA更新
            delta_W_q = delta_W_q + q_lora
            delta_W_v = delta_W_v + v_lora

        if self.pruning_mask is not None:
            # print("=================打印掩码========================")
            # print(self.pruning_mask.shape)
            # 拆分掩码为q、k、v部分
            dim = self.dim
            mask_q = self.pruning_mask[:dim, :]
            mask_v = self.pruning_mask[2*dim:, :]
            
            # 应用掩码到LoRA更新
            delta_W_q = delta_W_q * mask_q
            delta_W_v = delta_W_v * mask_v
        # 更新权重
        W_q_new = W_q + delta_W_q * self.s
        W_v_new = W_v + delta_W_v * self.s
        
        # 重新组合权重
        new_weight = torch.cat([W_q_new, W_k, W_v_new], dim=0)
        
        # 使用更新后的权重进行线性变换
        return F.linear(x, new_weight, self.bias)

    
class _LoRA_fc_timm(LoRABase):
    def __init__(self, fc: nn.Module, prune_rate: float, s: int = 1,block_idx=None, layer_name=None):
        super().__init__(block_idx=block_idx, layer_name=layer_name)
        self.fc = fc
        self.weight = fc.weight
        self.bias = fc.bias
        self.in_features = fc.in_features
        self.out_features = fc.out_features   
        self.prune_rate = prune_rate
        self.s = s
    
        self.initial_ranks = [8,8,8,8,8,8,8]
        self.extended_ranks = []
        self.active_ranks = []
        
        # 获取权重的数据类型
        weight_dtype = self.weight.dtype
        
        # 使用ParameterList存储LoRA参数
        self.fc_lora_down_list = nn.ParameterList()
        self.fc_lora_up_list = nn.ParameterList()
        
        for r in self.initial_ranks:
            # Down-Project层参数
            down_param = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.bfloat16))
            nn.init.zeros_(down_param)
            self.fc_lora_down_list.append(down_param)
            
            # Up-Project层参数
            up_param = nn.Parameter(torch.zeros(r, self.in_features, dtype=torch.bfloat16))
            nn.init.kaiming_uniform_(up_param, a=math.sqrt(5))
            self.fc_lora_up_list.append(up_param)

        self.active_ranks = self.initial_ranks.copy()
        
        self.weight.requires_grad = False
    def set_rank_requires_grad(self, target_indexes, requires_grad=True):   
        for idx in range(len(self.initial_ranks)):
            if idx in target_indexes:
                self.fc_lora_down_list[idx].requires_grad = requires_grad
                self.fc_lora_up_list[idx].requires_grad = requires_grad
            else:
                self.fc_lora_down_list[idx].requires_grad = False
                self.fc_lora_up_list[idx].requires_grad = False
    
    def forward(self, x):

        # 确保输入数据类型与权重一致
        if x.device != self.weight.device:
            print(f"设备不匹配: x在{x.device}, weight在{self.weight.device}")
            x = x.to(self.weight.device)
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        
        if not self.lora_enabled:
            return F.linear(x, self.weight, self.bias)
        
        # 获取原始权重
        weight = self.weight  # (out_features, in_features)
        
        # 初始化ΔW
        delta_W = torch.zeros_like(weight)
        
        # 应用激活的LoRA组
        for idx in self.get_active_indexes():
            lora_update = self.fc_lora_down_list[idx] @ self.fc_lora_up_list[idx]
            # 确保数据类型一致
            if lora_update.dtype != weight.dtype:
                lora_update = lora_update.to(weight.dtype)
            delta_W = delta_W + lora_update

        pruning_mask = self.get_pruning_mask(device=x.device)
        print(pruning_mask.shape)
        if pruning_mask is not None:
            # 确保掩码数据类型一致
            mask = pruning_mask.to(weight.device)
            if mask.dtype != weight.dtype:
                mask = mask.to(weight.dtype)
            delta_W = delta_W * mask
            # 更新权重: W_new = W_original + s * ΔW
            new_weight = weight * mask + self.s * delta_W
        
        # 确保新权重数据类型一致
        if new_weight.dtype != weight.dtype:
            new_weight = new_weight.to(weight.dtype)
        
        # 使用更新后的权重进行线性变换
        return F.linear(x, new_weight, self.bias)

class _LoRA_conv_timm(LoRABase):
    def __init__(self, conv_module, prune_rate=0.0, s=1):
        super().__init__()
        self.conv = conv_module
        for name, param in self.conv.named_parameters():
            self.register_parameter(name, param)
        self.prune_rate = prune_rate
        self.s = s
        self.in_channels = conv_module.in_channels
        self.out_channels = conv_module.out_channels
        # self.groups = conv_module.groups

        self.weight = self.conv.weight
        self.bias = self.conv.bias if hasattr(self.conv, 'bias') else None
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        kernel_size = conv_module.kernel_size
        if isinstance(kernel_size, tuple):
            # 假设宽高相同
            self.kernel_size = kernel_size[0]
        else:
            self.kernel_size = kernel_size

        # 初始化 LoRA 秩
        self.initial_ranks = [4,6,8,10]
        self.extended_ranks = []
        self.active_ranks = []
        self.conv_lora = nn.ParameterDict()
        for idx, r in enumerate(self.initial_ranks):
            # 计算输入和输出维度
            self.conv_lora[f"conv_lora_up_{idx}"] = nn.Parameter(
                torch.zeros(
                    r * self.kernel_size,
                    self.in_channels * self.kernel_size
                )
            ) 
            self.conv_lora[f"conv_lora_down_{idx}"] = nn.Parameter(
                torch.zeros(
                    self.out_channels * self.kernel_size,
                    r * self.kernel_size
                )
            )
            # 初始化参数
            nn.init.kaiming_uniform_(self.conv_lora[f"conv_lora_up_{idx}"], a=math.sqrt(5))
            nn.init.zeros_(self.conv_lora[f"conv_lora_down_{idx}"])

        self.active_ranks = self.initial_ranks.copy()

        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False
    
    def set_rank_requires_grad(self, target_ranks, requires_grad=True):
        for idx, r in enumerate(self.active_ranks):
            if r in target_ranks:
                self.conv_lora[f"conv_lora_down_{idx}"].requires_grad = requires_grad
                self.conv_lora[f"conv_lora_up_{idx}"].requires_grad = requires_grad
            else:
                self.conv_lora[f"conv_lora_down_{idx}"].requires_grad = False
                self.conv_lora[f"conv_lora_up_{idx}"].requires_grad = False
    def forward(self, x):
        device = x.device
        delta_weight = torch.zeros_like(self.conv.weight, device=device)
        if len(self.initial_ranks) > 0:
            for idx in self.get_active_indexes():
                # 获取 LoRA 调整
                lora_down = self.conv_lora[f"conv_lora_down_{idx}"].to(device)
                lora_up = self.conv_lora[f"conv_lora_up_{idx}"].to(device)
                
                lora_weight = lora_down @ lora_up
                lora_weight = lora_weight.view(
                    self.out_channels,
                    self.in_channels,
                    self.kernel_size,
                    self.kernel_size
                )
                delta_weight += lora_weight * self.s
        if self.pruning_mask is not None:
            delta_weight = delta_weight * self.pruning_mask
        new_weight = self.conv.weight + delta_weight
        bias = self.bias.to(device) if self.bias is not None else None
        return F.conv2d(
            x, 
            new_weight, 
            bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation,
            groups=self.groups
        )

   
def set_LoRA(layer=None,s=1,prune_rate=0.0,device=None,block_idx=None):
    # 适配resnet，mobilenet模型
    if isinstance(layer,BasicBlock):
        # 处理conv1
        print("找到了basicblock")
        original_conv1 = layer.conv1
        current_device = original_conv1.weight.device if device is None else device
        new_conv1 = _LoRA_conv_timm(original_conv1, prune_rate, s).to(current_device)
        layer.conv1 = new_conv1

        # 处理conv2
        original_conv2 = layer.conv2
        current_device = original_conv2.weight.device if device is None else device
        new_conv = _LoRA_conv_timm(original_conv2, prune_rate, s).to(current_device)
        layer.conv2 = new_conv
        # 处理downsample中的卷积层
        if layer.downsample is not None:
            if isinstance(layer.downsample, nn.Sequential):
                new_downsample = []
                for m in layer.downsample.children():
                    if isinstance(m, nn.Conv2d):
                        current_device = m.weight.device if device is None else device
                        new_downsample.append(_LoRA_conv_timm(m, prune_rate, s).to(current_device))
                    else:
                        new_downsample.append(m)
                layer.downsample = nn.Sequential(*new_downsample)
        elif isinstance(layer.downsample, nn.Conv2d):
            current_device = layer.downsample.weight.device if device is None else device
            layer.downsample = _LoRA_conv_timm(layer.downsample, prune_rate, s).to(current_device)
    elif isinstance(layer,Bottleneck):
        original_conv1 = layer.conv1
        current_device = original_conv1.weight.device if device is None else device
        new_conv1 = _LoRA_conv_timm(original_conv1, prune_rate, s).to(current_device)
        layer.conv1 = new_conv1

        # 处理conv2
        original_conv2 = layer.conv2
        current_device = original_conv2.weight.device if device is None else device
        new_conv = _LoRA_conv_timm(original_conv2, prune_rate, s).to(current_device)
        layer.conv2 = new_conv

        original_conv3 = layer.conv3
        current_device = original_conv3.weight.device if device is None else device
        new_conv = _LoRA_conv_timm(original_conv3, prune_rate, s).to(current_device)
        layer.conv3 = new_conv
        # 处理downsample中的卷积层
        if layer.downsample is not None:
            if isinstance(layer.downsample, nn.Sequential):
                new_downsample = []
                for m in layer.downsample.children():
                    if isinstance(m, nn.Conv2d):
                        current_device = m.weight.device if device is None else device
                        new_downsample.append(_LoRA_conv_timm(m, prune_rate, s).to(current_device))
                    else:
                        new_downsample.append(m)
                layer.downsample = nn.Sequential(*new_downsample)
            elif isinstance(layer.downsample, nn.Conv2d):
                current_device = layer.downsample.weight.device if device is None else device
                layer.downsample = _LoRA_conv_timm(layer.downsample, prune_rate, s).to(current_device)

    elif isinstance(layer,InvertedResidual):
        if hasattr(layer, 'conv') and isinstance(layer.conv, nn.Sequential):
            for i, sub_module in enumerate(layer.conv):
                # 处理 Conv2dNormActivation 中的卷积层
                if isinstance(sub_module, Conv2dNormActivation):
                    # 获取原始卷积层
                    original_conv = sub_module[0]
                    # 替换为 LoRA 卷积
                    current_device = original_conv.weight.device if device is None else device
                    new_conv = _LoRA_conv_timm(original_conv, prune_rate, s).to(current_device)
                    # 替换回 Conv2dNormActivation 中的卷积层
                    sub_module[0] = new_conv
                                        
                # 处理直接的 Conv2d 层
                elif isinstance(sub_module, nn.Conv2d):
                    # 直接替换
                    current_device = sub_module.weight.device if device is None else device
                    new_conv = _LoRA_conv_timm(sub_module, prune_rate, s).to(current_device)
                    layer.conv[i] = new_conv
                    
                # 处理 BatchNorm2d 层,保持不变
                elif isinstance(sub_module, nn.BatchNorm2d):
                    continue  
    # 适配llama模型
    if isinstance(layer,LlamaDecoderLayer):
        for name , _ in layer.named_children():
            if isinstance(_,LlamaAttention):
                current_device = _.q_proj.weight.device if device is None else device
                original_q_proj = _.q_proj
                new_q_proj = _LoRA_fc_timm(original_q_proj, prune_rate,s,block_idx=block_idx, layer_name=f"q_proj").to(current_device)
                _.q_proj = new_q_proj
                
                current_device = _.k_proj.weight.device if device is None else device
                original_k_proj = _.k_proj
                new_k_proj = _LoRA_fc_timm(original_k_proj, prune_rate,s,block_idx=block_idx, layer_name=f"k_proj").to(current_device)
                _.k_proj = new_k_proj

                current_device = _.v_proj.weight.device if device is None else device
                original_v_proj = _.v_proj
                new_v_proj = _LoRA_fc_timm(original_v_proj, prune_rate,s,block_idx=block_idx, layer_name=f"v_proj").to(current_device)
                _.v_proj = new_v_proj

                current_device = _.o_proj.weight.device if device is None else device
                original_o_proj = _.o_proj
                new_o_proj = _LoRA_fc_timm(original_o_proj, prune_rate,s,block_idx=block_idx, layer_name=f"o_proj").to(current_device)
                _.o_proj = new_o_proj
            elif isinstance(_,LlamaMLP):
                current_device = _.gate_proj.weight.device if device is None else device
                original_gate_proj = _.gate_proj
                new_gate_proj = _LoRA_fc_timm(original_gate_proj, prune_rate,s,block_idx=block_idx, layer_name=f"gate_proj").to(current_device)
                _.gate_proj = new_gate_proj

                current_device = _.up_proj.weight.device if device is None else device
                original_up_proj = _.up_proj
                new_up_proj = _LoRA_fc_timm(original_up_proj, prune_rate,s,block_idx=block_idx, layer_name=f"up_proj").to(current_device)
                _.up_proj = new_up_proj

                current_device = _.down_proj.weight.device if device is None else device
                original_down_proj = _.down_proj
                new_down_proj = _LoRA_fc_timm(original_down_proj, prune_rate,s,block_idx=block_idx, layer_name=f"down_proj").to(current_device)
                _.down_proj = new_down_proj
    if isinstance(layer, OPTDecoderLayer):
        for name, _ in layer.named_children():
            if isinstance(_, OPTAttention):
                # 处理注意力层中的投影
                current_device = _.q_proj.weight.device if device is None else device
                original_q_proj = _.q_proj
                new_q_proj = _LoRA_fc_timm(original_q_proj, prune_rate, s).to(current_device)
                _.q_proj = new_q_proj

                # # 处理 k_proj
                # current_device = _.k_proj.weight.device if device is None else device
                # original_k_proj = _.k_proj
                # new_k_proj = _LoRA_fc_timm(original_k_proj, prune_rate, s).to(current_device)
                # _.k_proj = new_k_proj

                current_device = _.v_proj.weight.device if device is None else device
                original_v_proj = _.v_proj
                new_v_proj = _LoRA_fc_timm(original_v_proj, prune_rate, s).to(current_device)
                _.v_proj = new_v_proj

                current_device = _.out_proj.weight.device if device is None else device
                original_out_proj = _.out_proj
                new_out_proj = _LoRA_fc_timm(original_out_proj, prune_rate, s).to(current_device)
                _.out_proj = new_out_proj
        
        if hasattr(layer, 'fc1'):
            current_device = layer.fc1.weight.device if device is None else device
            fc1 = layer.fc1
            new_fc1 = _LoRA_fc_timm(fc1, prune_rate, s).to(current_device)
            layer.fc1 = new_fc1

        if hasattr(layer, 'fc2'):
            current_device = layer.fc2.weight.device if device is None else device
            fc2 = layer.fc2
            new_fc2 = _LoRA_fc_timm(fc2, prune_rate, s).to(current_device)
            layer.fc2 = new_fc2

    for name,_ in layer.named_children():
        if isinstance(_,ViTSelfAttention):
            current_device = _.qkv.weight.device if device is None else device
            # 替换QKV层
            original_qkv = _.qkv
            new_qkv = _LoRA_qkv_timm(original_qkv, prune_rate,s).to(current_device)
            _.qkv = new_qkv
            
            original_porj = _.proj
            new_proj = _LoRA_fc_timm(original_porj, prune_rate,s).to(current_device)
            _.proj = new_proj
        elif isinstance(_,Mlp):
            # 替换mlp层
            current_device = _.fc1.weight.device if device is None else device
            original_fc1 = _.fc1
            new_fc1 = _LoRA_fc_timm(original_fc1, prune_rate,s).to(current_device)
            _.fc1 = new_fc1

            original_fc2 = _.fc2
            new_fc2 = _LoRA_fc_timm(original_fc2, prune_rate,s).to(current_device)
            _.fc2 = new_fc2
        elif len(list(_.children())) != 0:
            set_LoRA(_, prune_rate,s)

def set_lora_ranks_requires_grad(module, target_indexes, requires_grad=True):
    for m in module.modules():
        if isinstance(m, (_LoRA_qkv_timm, _LoRA_fc_timm, _LoRA_conv_timm)):
            print(f"找到了{m.__class__.__name__}")
            m.set_rank_requires_grad(target_indexes, requires_grad)
def set_lora_active_group_ranks(module, target_indexes):
    for m in module.modules():
        if isinstance(m, (LoRABase,)):
            m.set_active_group_ranks(target_indexes)
def set_lora_enabled(module, enabled=True):
    """
    设置LoRA模块的启用状态
    
    参数:
        module: 包含LoRA模块的层或模型
        enabled: 是否启用LoRA，True为启用，False为禁用
    """
    for m in module.modules():
        if hasattr(m, 'set_lora_enabled'):
            m.set_lora_enabled(enabled)