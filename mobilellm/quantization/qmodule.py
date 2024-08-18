import torch, math
import torch.nn as nn
from copy import deepcopy
from dataclasses import dataclass
from typing_extensions import Self
from mobilellm.model.hf_model import HFRMSNorm
from mobilellm.model.ops import FMatMul, L2Norm
from transformers.activations import GELUActivation, PytorchGELUTanh


CLIPMIN = 1e-5
CLIPMAX = 1e6


############################################################################
# Making quantization differentiable
def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


############################################################################
# Compute min and max from tensor
def compute_min_max_from_tensor(x, is_per_channel=False, group_size=-1):
    if is_per_channel:
        if group_size != -1:
            x = x.reshape(-1, group_size)
        min_val, max_val = torch.amin(x, dim=-1, keepdim=True), torch.amax(x, dim=-1, keepdim=True)
    else:
        y = x.view(-1)
        min_val, max_val = torch.amin(y, dim=-1), torch.amax(y, dim=-1)
    return min_val, max_val


############################################################################
# Compute scale and offset from min and max
# We use scale and offset as they are potentially learnable
def compute_scale_offset_from_min_max(min_val, max_val, bitwdith, is_symmetric):
    if not isinstance(min_val, torch.Tensor):
        min_val = torch.tensor(min_val)
    if not isinstance(max_val, torch.Tensor):
        max_val = torch.tensor(max_val)
    if is_symmetric:
        alpha = torch.maximum(torch.abs(min_val), torch.abs(max_val))
        beta  = 0
        q_min = -2 ** (bitwdith - 1)
        q_max = 2 ** (bitwdith - 1) - 1
    else:
        alpha = max_val - min_val
        beta = min_val
        q_min = 0
        q_max = (2 ** bitwdith - 1)
    # if alpha.max() < 0.01:
    #     alpha = torch.ones_like(alpha) * 0.01
    scale = alpha / q_max
    scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
    # offset = -((beta / alpha) * q_max).round()
    offset = -(beta / scale).round()
    return scale, offset, alpha, beta, q_min, q_max


############################################################################
# Compute min and max from scale and offset
def compute_min_max_from_scale_offset(scale, offset, bitwidth, is_symmetric):
    if is_symmetric:
        q_max = 2 ** (bitwidth - 1) - 1
    else:
        q_max = (2 ** bitwidth - 1)
    scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
    alpha = scale * q_max
    beta = -offset * scale
    max_val = alpha + beta
    min_val = beta if not is_symmetric else -max_val
    return min_val, max_val
    

############################################################################
# Meta data for quantization
@dataclass
class QuantConfig:
    bitwidth:       int  = 32
    group_size:     int  = -1
    is_symmetric:   bool = False
    is_per_channel: bool = False
    is_dynamic:     bool = False
        
    @classmethod
    def from_dict(cls, cfg: dict) -> Self:
        cfg_dict = {
            'bitwidth': int(cfg['bitwidth']),
            'group_size': int(cfg['group_size']),
            'is_symmetric': cfg['is_symmetric'] in ["True", "true"],
            'is_per_channel': cfg['is_per_channel'] in ["True", "true"],
            'is_dynamic': cfg['is_dynamic'] in ["True", "true"],
        }
        return cls(**cfg_dict)
    
    def to_dict(self):
        return {
            'bitwidth': str(self.bitwidth),
            'group_size': str(self.group_size),
            'is_symmetric': str(self.is_symmetric),
            'is_per_channel': str(self.is_per_channel),
            'is_dynamic': str(self.is_dynamic),
        }


############################################################################
# Differentiable quantizer
class Quantizer(nn.Module):
    def __init__(self, qcfg):
        super(Quantizer, self).__init__()
        self.qcfg = deepcopy(qcfg)
        self.lwc = False
        self.enable = True 

    def update_qcfg(self, qcfg):
        if not isinstance(qcfg, QuantConfig):
            assert(isinstance(qcfg, dict))
            qcfg = QuantConfig.from_dict(qcfg)

        self.qcfg = deepcopy(qcfg)
        if hasattr(self, 'scale'):
            del self.scale
        if hasattr(self, 'offset'):
            del self.offset

    def export_qcfg(self):
        return self.qcfg.to_dict()

    def enable_lwc(self, w):
        self.lwc = True
        init_value = 4.
        shape = w.shape
        if self.qcfg.group_size != -1:
            dim1 = int(shape[0] * math.ceil(shape[1] / self.qcfg.group_size))
            self.deficiency = shape[-1] % group_size
            if self.deficiency > 0:
                self.deficiency = group_size - self.deficiency
                assert self.qcfg.is_symmetric   # support for mlc-llm symmetric quantization
        else:
            dim1 = shape[0]
        if not self.qcfg.is_per_channel:
            # scalar bound
            self.upbound_factor  = nn.Parameter(torch.ones(1, dtype=w.dtype, device=w.device) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones(1, dtype=w.dtype, device=w.device) * init_value)
        else:
            self.upbound_factor  = nn.Parameter(torch.ones((dim1, 1), dtype=w.dtype, device=w.device) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1, 1), dtype=w.dtype, device=w.device) * init_value)

    def disable_lwc(self,):
        if self.lwc:
            self.lwc = False
        if hasattr(self, "upbound_factor"): del self.upbound_factor
        if hasattr(self, "lowbound_factor"): del self.lowbound_factor

    def run_lwc(self, input_):
        x = input_
        dtype = input_.dtype
        shape = input_.shape

        if self.qcfg.is_per_channel and self.qcfg.group_size != -1:
            x = x.reshape(-1, self.qcfg.group_size)

        if self.qcfg.is_per_channel:
            min_val, max_val = torch.amin(x, dim=-1, keepdim=True), torch.amax(x, dim=-1, keepdim=True)
        else:
            # min_val, max_val = torch.min(x), torch.max(x)
            y = x.contiguous().view(-1)
            min_val, max_val = torch.amin(y, dim=-1), torch.amax(y, dim=-1)
        if self.lwc:
            max_val = torch.nn.functional.sigmoid(self.upbound_factor) * max_val
            min_val = torch.nn.functional.sigmoid(self.lowbound_factor) * min_val
            # self.set_scale_offset_from_minmax(min_val, max_val, None, x.device)
            if hasattr(self, "scale"):
                del self.scale
                del self.offset
            self.disable_lwc()
        x = x.clamp(min_val, max_val)

        if self.qcfg.is_per_channel and self.qcfg.group_size != -1:
            x = x.reshape(shape)
        return x.type(dtype)

    # def run_clipping(self, input_):
    #     x = input_
    #     dtype = input_.dtype
    #     shape = input_.shape

    #     if self.qcfg.is_per_channel and self.qcfg.group_size != -1:
    #         x = x.reshape(-1, self.qcfg.group_size)

    #     if self.qcfg.is_per_channel:
    #         if hasattr(self, "scale"):
    #             min_val, max_val = compute_min_max_from_scale_offset(self.scale, self.offset, self.qcfg.bitwidth, self.qcfg.is_symmetric)
    #         else:
    #             min_val, max_val = torch.amin(x, dim=-1, keepdim=True), torch.amax(x, dim=-1, keepdim=True)
    #     else:
    #         if hasattr(self, "scale"):
    #             min_val, max_val = compute_min_max_from_scale_offset(self.scale, self.offset, self.qcfg.bitwidth, self.qcfg.is_symmetric)
    #         else:
    #             y = x.view(-1)
    #             min_val, max_val = torch.amin(y, dim=-1), torch.amax(y, dim=-1)
    #     if self.lwc:
    #         max_val = torch.nn.functional.sigmoid(self.upbound_factor) * max_val
    #         min_val = torch.nn.functional.sigmoid(self.lowbound_factor) * min_val
    #         self.disable_lwc()
    #     x = x.clamp(min_val, max_val)

    #     if self.qcfg.is_per_channel and self.qcfg.group_size != -1:
    #         x = x.reshape(shape)
    #     return x.type(dtype)

    def set_scale_offset_from_minmax(self, min_val, max_val, cache_mode=None, device=None):
        scale, offset, alpha, beta, q_min, q_max = \
            compute_scale_offset_from_min_max(
                min_val, 
                max_val, 
                self.qcfg.bitwidth, 
                self.qcfg.is_symmetric
            )
        self.qmin = q_min
        self.qmax = q_max
        if cache_mode == "parameter":
            if torch.is_tensor(scale):
                self.register_parameter('scale', nn.Parameter(scale.to(device)))
                self.register_parameter('offset', nn.Parameter(offset.to(device)))
            else:
                self.register_parameter('scale', nn.Parameter(torch.tensor(scale, device=device)))
                self.register_parameter('offset', nn.Parameter(torch.tensor(offset, device=device)))
        elif cache_mode == "buffer":
            if torch.is_tensor(scale):
                self.register_buffer('scale', scale.to(device))
                self.register_buffer('offset', offset.to(device))
            else:
                self.register_buffer('scale', torch.tensor(scale, device=device))
                self.register_buffer('offset', torch.tensor(offset, device=device))
        else:
            if not torch.is_tensor(scale):
                scale = torch.tensor(scale, device=device)
                offset = torch.tensor(offset, device=device)
            self.scale = scale
            self.offset = offset
    
    def set_scale_offset_from_tensor(self, x, cache_mode=None):
        min_val, max_val = compute_min_max_from_tensor(x, self.qcfg.is_per_channel, self.qcfg.group_size)
        self.set_scale_offset_from_minmax(min_val, max_val, cache_mode, x.device)

    def forward(self, input_, use_scale_offset_as="parameter"):
        if not self.enable or self.qcfg.bitwidth > 16:
            return input_

        x = input_
        dtype = input_.dtype
        shape = input_.shape

        if self.qcfg.is_per_channel and self.qcfg.group_size != -1:
            x = x.reshape(-1, self.qcfg.group_size)
        
        if self.qcfg.is_dynamic or self.lwc or (not hasattr(self, 'scale')) or (not hasattr(self, 'offset')):
            if self.qcfg.is_per_channel:
                min_val, max_val = torch.amin(x, dim=-1, keepdim=True), torch.amax(x, dim=-1, keepdim=True)
            else:
                # min_val, max_val = torch.min(x), torch.max(x)
                y = x.contiguous().view(-1)
                min_val, max_val = torch.amin(y, dim=-1), torch.amax(y, dim=-1)

            if self.lwc:
                max_val = torch.nn.functional.sigmoid(self.upbound_factor) * max_val
                min_val = torch.nn.functional.sigmoid(self.lowbound_factor) * min_val

            if self.qcfg.is_dynamic or self.lwc:
                self.set_scale_offset_from_minmax(min_val, max_val, None, x.device)
            else:
                self.set_scale_offset_from_minmax(min_val, max_val, use_scale_offset_as, x.device)

        if self.scale.device != x.device:
            self.scale.data = self.scale.to(x.device)

        if self.offset.device != x.device:
            self.offset.data = self.offset.to(x.device)
                    
        # quantize
        x = round_ste(x / self.scale) + self.offset
        x = x.clamp(self.qmin, self.qmax)

        # dequantize
        x = (x - self.offset) * self.scale

        if self.qcfg.is_per_channel and self.qcfg.group_size != -1:
            x = x.reshape(shape)

        return x.type(dtype)
        

class QLinear(nn.Linear):
    def __init__(self, kargs, input_quant_cfg, weight_quant_cfg, output_quant_cfg):
        super(QLinear, self).__init__(**kargs)

        self.input_quantizer  = None
        self.output_quantizer = None
        self.weight_quantizer = None
        self.use_temporary_parameter = False

        if weight_quant_cfg is not None:
            self.weight_quantizer = Quantizer(weight_quant_cfg)
        if input_quant_cfg is not None:
            self.input_quantizer = Quantizer(input_quant_cfg)
        if output_quant_cfg is not None:
            self.output_quantizer = Quantizer(output_quant_cfg)

    def update_qcfg(self, input_quant_cfg, weight_quant_cfg, output_quant_cfg):
        if self.input_quantizer is not None:
            self.input_quantizer.update_qcfg(input_quant_cfg)
        if self.weight_quantizer is not None:
            self.weight_quantizer.update_qcfg(weight_quant_cfg)
        if self.output_quantizer is not None:
            self.output_quantizer.update_qcfg(output_quant_cfg)
        
    def export_qcfg(self):
        res = {}
        if self.input_quantizer is not None:
            res["input"] = self.input_quantizer.export_qcfg()
        if self.weight_quantizer is not None:
            res["weight"] = self.weight_quantizer.export_qcfg()
        if self.output_quantizer is not None:
            res["output"] = self.output_quantizer.export_qcfg()
        return res
    
    def set_scale_offset(self, act_scale, use_scale_offset_as="parameter"):
        # Note that we do not set the scale and offset for the weight 
        # as the weight is a single matrix. Its stats will be computed on-the-fly
        if self.input_quantizer is not None:
            self.input_quantizer.set_scale_offset_from_minmax(act_scale['input'][0], act_scale['input'][1], use_scale_offset_as, self.weight.device)
        
        if self.output_quantizer is not None:
            self.output_quantizer.set_scale_offset_from_minmax(act_scale['output'][0], act_scale['output'][1], use_scale_offset_as, self.weight.device)

    def forward(self, input_):
        # quantize weight
        weight = self.weight if not self.use_temporary_parameter else self.temp_weight
        bias = self.bias if not self.use_temporary_parameter else self.temp_bias

        if self.weight_quantizer is not None:
            weight = self.weight_quantizer(weight)

        # quantize inputs
        if self.input_quantizer is not None:
            input_ = self.input_quantizer(input_)

        out = nn.functional.linear(input_, weight, bias=bias)

        # quantize output
        if self.output_quantizer is not None:
            out = self.output_quantizer(out)
        return out

    @staticmethod
    def from_float(module, input_quant_cfg, weight_quant_cfg, output_quant_cfg):
        kargs = {
            'in_features': module.in_features, 
            'out_features': module.out_features, 
            'bias': module.bias is not None,
            'device': module.weight.device,
            'dtype': module.weight.dtype,
        }
        out = QLinear(
            kargs,
            input_quant_cfg=input_quant_cfg,
            weight_quant_cfg=weight_quant_cfg,
            output_quant_cfg=output_quant_cfg
        )
        with torch.no_grad():
            out.weight.copy_(module.weight)
            if out.bias is not None:
                out.bias.copy_(module.bias)
        return out
    
    @staticmethod
    def to_float(module):
        kargs = {
            'in_features': module.in_features, 
            'out_features': module.out_features, 
            'bias': module.bias is not None,
            'device': module.weight.device,
            'dtype': module.weight.dtype,
        }
        out = nn.Linear(**kargs)
        with torch.no_grad():
            weight = module.weight
            # if module.weight_quantizer is not None:
            #     # temporary hack
            #     if hasattr(module.weight_quantizer, "scale"):
            #         del module.weight_quantizer.scale
            #         del module.weight_quantizer.offset
            #     weight = module.weight_quantizer.run_clipping(weight)
            out.weight.copy_(weight)
            if module.bias is not None:
                if out.bias is None:
                    out.bias = module.bias
                else:
                    out.bias.copy_(module.bias)
        return out


class QMatMul(nn.Module):
    def __init__(self, input_quant_cfg, input2_quant_cfg, output_quant_cfg):
        super().__init__()

        self.input_quantizer = None
        self.input2_quantizer = None
        self.output_quantizer = None

        if input_quant_cfg is not None:
            self.input_quantizer = Quantizer(input_quant_cfg)

        if input2_quant_cfg is not None:
            self.input2_quantizer = Quantizer(input2_quant_cfg)

        if output_quant_cfg is not None:
            self.output_quantizer = Quantizer(output_quant_cfg)
    
    def update_qcfg(self, input_quant_cfg, input2_quant_cfg, output_quant_cfg):
        if self.input_quantizer is not None:
            self.input_quantizer.update_qcfg(input_quant_cfg)
        if self.input2_quantizer is not None:
            self.input2_quantizer.update_qcfg(input2_quant_cfg)
        if self.output_quantizer is not None:
            self.output_quantizer.update_qcfg(output_quant_cfg)

    def export_qcfg(self):
        res = {}
        if self.input_quantizer is not None:
            res["input"] = self.input_quantizer.export_qcfg()
        if self.input2_quantizer is not None:
            res["input2"] = self.input2_quantizer.export_qcfg()
        if self.output_quantizer is not None:
            res["output"] = self.output_quantizer.export_qcfg()
        return res

    def set_scale_offset(self, act_scale, use_scale_offset_as="parameter"):
        if self.input_quantizer is not None:
            self.input_quantizer.set_scale_offset_from_minmax(act_scale['input'][0], act_scale['input'][1], use_scale_offset_as)

        if self.input2_quantizer is not None:
            self.input2_quantizer.set_scale_offset_from_minmax(act_scale['input2'][0], act_scale['input2'][1], use_scale_offset_as)
        
        if self.output_quantizer is not None:
            self.output_quantizer.set_scale_offset_from_minmax(act_scale['output'][0], act_scale['output'][1], use_scale_offset_as)

    def forward(self, x1, x2):
        # quantize input
        if self.input_quantizer is not None:
            x1 = self.input_quantizer(x1)

        if self.input2_quantizer is not None:
            x2 = self.input2_quantizer(x2)
        
        out = torch.matmul(x1, x2)
        
        # quantize output
        if self.output_quantizer is not None:
            out = self.output_quantizer(out)
        return out


class QRMSNorm(HFRMSNorm):
    def __init__(self, kargs, input_quant_cfg, weight_quant_cfg, output_quant_cfg):
        super(QRMSNorm, self).__init__(**kargs)

        self.input_quantizer  = None
        self.weight_quantizer = None
        self.output_quantizer = None
        self.use_temporary_parameter = False

        if input_quant_cfg is not None:
            self.input_quantizer = Quantizer(input_quant_cfg)
            # # lets disable the input quantizer by default to avoid duplication, as the input may be an quantized output from the previous layer
            # self.input_quantizer.enable = False
        if weight_quant_cfg is not None:
            self.weight_quantizer = Quantizer(weight_quant_cfg)
        if output_quant_cfg is not None:
            self.output_quantizer = Quantizer(output_quant_cfg)

    def update_qcfg(self, input_quant_cfg, weight_quant_cfg, output_quant_cfg):
        if self.input_quantizer is not None:
            self.input_quantizer.update_qcfg(input_quant_cfg)
        if self.weight_quantizer is not None:
            self.weight_quantizer.update_qcfg(weight_quant_cfg)
        if self.output_quantizer is not None:
            self.output_quantizer.update_qcfg(output_quant_cfg)
        
    def export_qcfg(self):
        res = {}
        if self.input_quantizer is not None:
            res["input"] = self.input_quantizer.export_qcfg()
        if self.weight_quantizer is not None:
            res["weight"] = self.weight_quantizer.export_qcfg()
        if self.output_quantizer is not None:
            res["output"] = self.output_quantizer.export_qcfg()
        return res

    def set_scale_offset(self, act_scale, use_scale_offset_as="parameter"):
        if self.input_quantizer is not None:
            self.input_quantizer.set_scale_offset_from_minmax(act_scale['input'][0], act_scale['input'][1], use_scale_offset_as, self.weight.device)
        
        # if self.weight_quantizer is not None:
        #     self.weight_quantizer.set_scale_offset_from_tensor(self.weight, use_scale_offset_as)

        if self.output_quantizer is not None:
            self.output_quantizer.set_scale_offset_from_minmax(act_scale['output'][0], act_scale['output'][1], use_scale_offset_as, self.weight.device)

    def forward(self, input_):
        # quantize weight
        weight = self.weight if not self.use_temporary_parameter else self.temp_weight
        if self.weight_quantizer is not None:
            weight = self.weight_quantizer(weight)

        # quantize inputs
        if self.input_quantizer is not None:
            input_ = self.input_quantizer(input_)

        # out = rms_norm(input_, weight, self.eps, self.bias)
        out = self.forward_impl(input_, weight, self.bias)

        # quantize output
        if self.output_quantizer is not None:
            out = self.output_quantizer(out)
        return out

    @staticmethod
    def from_float(module, input_quant_cfg, weight_quant_cfg, output_quant_cfg):
        kargs = {
            'dim': len(module.weight), 
            'eps': module.eps,
            'device': module.weight.device,
            'dtype': module.weight.dtype,
            'l2norm_as_rmsnorm': module.l2norm_as_rmsnorm,
            # 'bias': module.bias is not None,
        }
        out = QRMSNorm(
            kargs,
            input_quant_cfg=input_quant_cfg,
            weight_quant_cfg=weight_quant_cfg,
            output_quant_cfg=output_quant_cfg
        )
        with torch.no_grad():
            out.weight.copy_(module.weight)
            # if out.bias is not None:
            #     out.bias.copy_(module.bias)
        return out
    
    @staticmethod
    def to_float(module):
        kargs = {
            'dim': len(module.weight), 
            'eps': module.eps,
            'device': module.weight.device,
            'dtype': module.weight.dtype,
            'l2norm_as_rmsnorm': module.l2norm_as_rmsnorm,
            # 'bias': module.bias is not None,
        }
        out = HFRMSNorm(**kargs)
        with torch.no_grad():
            weight = module.weight
            # if module.weight_quantizer is not None:
            #     weight = module.weight_quantizer.run_clipping(weight)
            out.weight.copy_(weight)
            # if module.bias is not None:
            #     if out.bias is None:
            #         out.bias = module.bias
            #     else:
            #         out.bias.copy_(module.bias)
        return out


class QLayerNorm(nn.LayerNorm):
    def __init__(self, kargs, input_quant_cfg, weight_quant_cfg, output_quant_cfg):
        super(QLayerNorm, self).__init__(**kargs)

        self.input_quantizer  = None
        self.output_quantizer = None
        self.weight_quantizer = None
        self.use_temporary_parameter = False

        if weight_quant_cfg is not None:
            self.weight_quantizer = Quantizer(weight_quant_cfg)
        if input_quant_cfg is not None:
            self.input_quantizer = Quantizer(input_quant_cfg)
            # # lets disable the input quantizer by default to avoid duplication, as the input may be quantized from the previous layer
            # self.input_quantizer.enable = False
        if output_quant_cfg is not None:
            self.output_quantizer = Quantizer(output_quant_cfg)

    def update_qcfg(self, input_quant_cfg, weight_quant_cfg, output_quant_cfg):
        if self.input_quantizer is not None:
            self.input_quantizer.update_qcfg(input_quant_cfg)
        if self.weight_quantizer is not None:
            self.weight_quantizer.update_qcfg(weight_quant_cfg)
        if self.output_quantizer is not None:
            self.output_quantizer.update_qcfg(output_quant_cfg)
        
    def export_qcfg(self):
        res = {}
        if self.input_quantizer is not None:
            res["input"] = self.input_quantizer.export_qcfg()
        if self.weight_quantizer is not None:
            res["weight"] = self.weight_quantizer.export_qcfg()
        if self.output_quantizer is not None:
            res["output"] = self.output_quantizer.export_qcfg()
        return res
    
    def set_scale_offset(self, act_scale, use_scale_offset_as="parameter"):
        if self.input_quantizer is not None:
            self.input_quantizer.set_scale_offset_from_minmax(act_scale['input'][0], act_scale['input'][1], use_scale_offset_as, self.weight.device)
        
        # if self.weight_quantizer is not None:
        #     self.weight_quantizer.set_scale_offset_from_tensor(self.weight, use_scale_offset_as)

        if self.output_quantizer is not None:
            self.output_quantizer.set_scale_offset_from_minmax(act_scale['output'][0], act_scale['output'][1], use_scale_offset_as, self.weight.device)

    def forward(self, input_):
        # quantize weight
        weight = self.weight if not self.use_temporary_parameter else self.temp_weight
        bias = self.bias if not self.use_temporary_parameter else self.temp_bias

        if self.weight_quantizer is not None:
            weight = self.weight_quantizer(weight)

        # quantize inputs
        if self.input_quantizer is not None:
            input_ = self.input_quantizer(input_)

        out = nn.functional.layer_norm(input_, input_.shape[-1:], weight=weight, bias=bias, eps=self.eps)

        # quantize output
        if self.output_quantizer is not None:
            out = self.output_quantizer(out)
        return out

    @staticmethod
    def from_float(module, input_quant_cfg, weight_quant_cfg, output_quant_cfg):
        kargs = {
            'normalized_shape': len(module.weight), 
            'eps': module.eps,
            'elementwise_affine': module.elementwise_affine,
            # 'bias': module.bias is not None,
            'device': module.weight.device,
            'dtype': module.weight.dtype,
        }
        out = QLayerNorm(
            kargs,
            input_quant_cfg=input_quant_cfg,
            weight_quant_cfg=weight_quant_cfg,
            output_quant_cfg=output_quant_cfg
        )
        with torch.no_grad():
            weight = module.weight
            out.weight.copy_(weight)
            if out.bias is not None:
                out.bias.copy_(module.bias)
        return out
    
    @staticmethod
    def to_float(module):
        kargs = {
            'normalized_shape': len(module.weight), 
            'eps': module.eps,
            'elementwise_affine': module.elementwise_affine,
            'bias': module.bias is not None,
            'device': module.weight.device,
            'dtype': module.weight.dtype,
        }
        out = nn.LayerNorm(**kargs)
        with torch.no_grad():
            weight = module.weight
            # if module.weight_quantizer is not None:
            #     weight = module.weight_quantizer.run_clipping(weight)
            out.weight.copy_(weight)
            if module.bias is not None:
                if out.bias is None:
                    out.bias = module.bias
                else:
                    out.bias.copy_(module.bias)
        return out


class QSiLU(nn.Module):
    def __init__(self, input_quant_cfg, input2_quant_cfg, output_quant_cfg):
        super(QSiLU, self).__init__()

        self.input_quantizer = None
        self.input2_quantizer = None
        self.output_quantizer = None

        if input_quant_cfg is not None:
            self.input_quantizer = Quantizer(input_quant_cfg)

        if input2_quant_cfg is not None:
            self.input2_quantizer = Quantizer(input2_quant_cfg)

        if output_quant_cfg is not None:
            self.output_quantizer = Quantizer(output_quant_cfg)

    def update_qcfg(self, input_quant_cfg, input2_quant_cfg, output_quant_cfg):
        if self.input_quantizer is not None and input_quant_cfg is not None:
            self.input_quantizer.update_qcfg(input_quant_cfg)
        if self.input2_quantizer is not None:
            self.input2_quantizer.update_qcfg(input2_quant_cfg)
        if self.output_quantizer is not None:
            self.output_quantizer.update_qcfg(output_quant_cfg)

    def export_qcfg(self):
        res = {}
        if self.input_quantizer is not None:
            res["input"] = self.input_quantizer.export_qcfg()
        if self.input2_quantizer is not None:
            res["input2"] = self.input2_quantizer.export_qcfg()
        if self.output_quantizer is not None:
            res["output"] = self.output_quantizer.export_qcfg()
        return res

    def set_scale_offset(self, act_scale, use_scale_offset_as="parameter"):
        if self.input_quantizer is not None:
            self.input_quantizer.set_scale_offset_from_minmax(act_scale['input'][0], act_scale['input'][1], use_scale_offset_as)

        if self.input2_quantizer is not None:
            if "input2" in act_scale:
                self.input2_quantizer.set_scale_offset_from_minmax(act_scale['input2'][0], act_scale['input2'][1], use_scale_offset_as)
            else:
                self.input2_quantizer.set_scale_offset_from_minmax(0.0, 1.0, use_scale_offset_as)

        if self.output_quantizer is not None:
            self.output_quantizer.set_scale_offset_from_minmax(act_scale['output'][0], act_scale['output'][1], use_scale_offset_as)

    def forward(self, x):
        if self.input_quantizer is not None:
            x = self.input_quantizer(x)
        
        y = nn.functional.sigmoid(x)

        if self.input2_quantizer is not None:
            y = self.input2_quantizer(y)
        
        out = x * y
        
        # quantize output
        if self.output_quantizer is not None:
            out = self.output_quantizer(out)
        return out


class QGELU(nn.Module):
    def __init__(self, input_quant_cfg, output_quant_cfg):
        super(QGELU, self).__init__()

        self.input_quantizer = None
        self.output_quantizer = None

        if input_quant_cfg is not None:
            self.input_quantizer = Quantizer(input_quant_cfg)

        if output_quant_cfg is not None:
            self.output_quantizer = Quantizer(output_quant_cfg)
    
    def update_qcfg(self, input_quant_cfg, output_quant_cfg):
        if self.input_quantizer is not None and input_quant_cfg is not None:
            self.input_quantizer.update_qcfg(input_quant_cfg)
        if self.output_quantizer is not None:
            self.output_quantizer.update_qcfg(output_quant_cfg)

    def export_qcfg(self):
        res = {}
        if self.input_quantizer is not None:
            res["input"] = self.input_quantizer.export_qcfg()
        if self.output_quantizer is not None:
            res["output"] = self.output_quantizer.export_qcfg()
        return res

    def set_scale_offset(self, act_scale, use_scale_offset_as="parameter"):
        if self.input_quantizer is not None:
            self.input_quantizer.set_scale_offset_from_minmax(act_scale['input'][0], act_scale['input'][1], use_scale_offset_as)

        if self.output_quantizer is not None:
            self.output_quantizer.set_scale_offset_from_minmax(act_scale['output'][0], act_scale['output'][1], use_scale_offset_as)

    def forward(self, x):
        if self.input_quantizer is not None:
            x = self.input_quantizer(x)
        
        out = nn.functional.gelu(x)
        
        # quantize output
        if self.output_quantizer is not None:
            out = self.output_quantizer(out)
        return out



def to_weight_only_quantized_linear(module, w_qcfg=None, w_scale=None, w_offset=None):
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
    if isinstance(module, QLinear):
        assert(module.weight_quantizer.qcfg.bitwidth in [2, 3, 4])
        assert(module.input_quantizer is None or module.input_quantizer.qcfg.bitwidth >= 16)
        assert(module.output_quantizer is None or module.output_quantizer.qcfg.bitwidth >= 16)
        if w_qcfg is None: w_qcfg = module.weight_quantizer.qcfg
        if w_scale is None: w_scale = getattr(module.weight_quantizer, "scale", None)
        if w_offset is None: w_offset = getattr(module.weight_quantizer, "offset", None)
    
    if w_scale is None or w_offset is None:
        min_val, max_val = compute_min_max_from_tensor(module.weight.to(torch.float32), w_qcfg.is_per_channel, w_qcfg.group_size)
        w_scale, w_offset, *_ = compute_scale_offset_from_min_max(min_val, max_val, w_qcfg.bitwidth, w_qcfg.is_symmetric)

    group_size = w_qcfg.group_size
    out_features, in_features = module.weight.shape
    if group_size == -1:
        group_size = in_features
    assert(in_features % group_size == 0)
    n_groups = in_features // group_size
    scale_reshaped = w_scale.detach().data.reshape(out_features, n_groups)
    offset_reshaped = w_offset.detach().data.reshape(out_features, n_groups)
    device = module.weight.device
    out = qlinear_cuda.QuantLinear(w_qcfg.bitwidth, group_size, in_features, out_features, not module.bias is None, kernel_switch_threshold=128)
    out.pack(deepcopy(module).to("cpu"), scale_reshaped.to("cpu"), offset_reshaped.to("cpu"), g_idx=None)
    return out.to(device)


################################################################################################################
# For quantization

def create_sim_qmodel(model, default_weight_qcfg=None, default_act_qcfg=None):
    if default_weight_qcfg is None:
        default_weight_qcfg = QuantConfig()

    if default_act_qcfg is None:
        default_act_qcfg = QuantConfig()

    for name, module in reversed(model._modules.items()):
        if "lm_head" in name or ("norm" in name and "layernorm" not in name):
            # not quantizing the last norm and predictor at the moment
            pass
        elif isinstance(module, nn.Linear):
            model._modules[name] = QLinear.from_float(model._modules[name], default_act_qcfg, default_weight_qcfg, default_act_qcfg)
            if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name or "w1" in name or "w3" in name:
                # they have been quantized in previous layers
                model._modules[name].input_quantizer = None
        elif isinstance(module, FMatMul):
            model._modules[name] = QMatMul(default_act_qcfg, default_act_qcfg, default_act_qcfg)
        elif isinstance(module, nn.SiLU):
            model._modules[name] = QSiLU(default_act_qcfg, default_act_qcfg, default_act_qcfg)
            model._modules[name].input_quantizer = None
        elif isinstance(module, (nn.GELU, GELUActivation, PytorchGELUTanh)):
            model._modules[name] = QGELU(default_act_qcfg, default_act_qcfg)
            model._modules[name].input_quantizer = None
        elif isinstance(module, HFRMSNorm):
            model._modules[name] = QRMSNorm.from_float(model._modules[name], default_act_qcfg, default_weight_qcfg, default_act_qcfg)
        elif isinstance(module, nn.LayerNorm):
            model._modules[name] = QLayerNorm.from_float(model._modules[name], default_act_qcfg, default_weight_qcfg, default_act_qcfg)
        elif len(list(module.children())) > 0:
            create_sim_qmodel(module, default_weight_qcfg, default_act_qcfg)
    return model


def create_weight_only_qmodel(model, w_qcfg=None):
    for name, module in reversed(model._modules.items()):
        if isinstance(module, QLinear):
            model._modules[name] = to_weight_only_quantized_linear(model._modules[name])
        elif isinstance(module, QRMSNorm):
            model._modules[name] = QRMSNorm.to_float(model._modules[name])
        elif isinstance(module, QLayerNorm):
            model._modules[name] = QLayerNorm.to_float(model._modules[name])
        elif isinstance(module, QMatMul):
            model._modules[name] = FMatMul()
        elif isinstance(module, QSiLU):
            model._modules[name] = nn.SiLU()
        elif isinstance(module, QGELU):
            model._modules[name] = nn.GELU()
        elif isinstance(module, nn.Linear) and 'lm_head' not in name:
            model._modules[name] = to_weight_only_quantized_linear(model._modules[name], w_qcfg)
        elif len(list(module.children())) > 1:
            create_weight_only_qmodel(module, w_qcfg)
    return model


def create_fp_model(model):
    for name, module in reversed(model._modules.items()):
        if isinstance(module, QLinear):
            model._modules[name] = QLinear.to_float(model._modules[name])
        elif isinstance(module, QRMSNorm):
            model._modules[name] = QRMSNorm.to_float(model._modules[name])
        elif isinstance(module, QLayerNorm):
            model._modules[name] = QLayerNorm.to_float(model._modules[name])
        elif isinstance(module, QMatMul):
            model._modules[name] = FMatMul()
        elif isinstance(module, QSiLU):
            model._modules[name] = nn.SiLU()
        elif isinstance(module, QGELU):
            model._modules[name] = nn.GELU()
        elif len(list(module.children())) > 1:
            create_fp_model(module)
    return model


def export_act_range(model):
    act_dict = {}
    for name, m in model.named_modules():
        if isinstance(m, (QLinear, QRMSNorm, QLayerNorm, QGELU)):
            entry = act_dict.get(name, {})
            if m.input_quantizer is not None:
                qr = m.input_quantizer
                min_val, max_val = compute_min_max_from_scale_offset(qr.scale, qr.offset, qr.qcfg.bitwidth, qr.qcfg.is_symmetric)
                entry["input"] = [min_val.item(), max_val.item()]
            if m.output_quantizer is not None:
                qr = m.output_quantizer
                min_val, max_val = compute_min_max_from_scale_offset(qr.scale, qr.offset, qr.qcfg.bitwidth, qr.qcfg.is_symmetric)
                entry["output"] = [min_val.item(), max_val.item()]
            act_dict[name] = entry
        elif isinstance(m, (QMatMul, QSiLU)):
            entry = act_dict.get(name, {})
            if m.input_quantizer is not None:
                qr = m.input_quantizer
                min_val, max_val = compute_min_max_from_scale_offset(qr.scale, qr.offset, qr.qcfg.bitwidth, qr.qcfg.is_symmetric)
                entry["input"] = [min_val.item(), max_val.item()]
            if m.input2_quantizer is not None:
                qr = m.input2_quantizer
                min_val, max_val = compute_min_max_from_scale_offset(qr.scale, qr.offset, qr.qcfg.bitwidth, qr.qcfg.is_symmetric)
                entry["input2"] = [min_val.item(), max_val.item()]
            if m.output_quantizer is not None:
                qr = m.output_quantizer
                min_val, max_val = compute_min_max_from_scale_offset(qr.scale, qr.offset, qr.qcfg.bitwidth, qr.qcfg.is_symmetric)
                entry["output"] = [min_val.item(), max_val.item()]
            act_dict[name] = entry
    return act_dict


def update_qcfg(model, override_qcfg):
    for name, module in model.named_modules():
        if isinstance(module, (QLinear, QRMSNorm, QLayerNorm)):
            assert(name in override_qcfg)
            module.update_qcfg(override_qcfg[name].get("input", None), override_qcfg[name]["weight"], override_qcfg[name]["output"])
        elif isinstance(module, (QMatMul, )):
            assert(name in override_qcfg)
            module.update_qcfg(override_qcfg[name]["input"], override_qcfg[name]["input2"], override_qcfg[name]["output"])
        elif isinstance(module, (QSiLU, )):
            assert(name in override_qcfg)
            module.update_qcfg(override_qcfg[name].get("input", None), override_qcfg[name]["input2"], override_qcfg[name]["output"])
        elif isinstance(module, (QGELU, )):
            assert(name in override_qcfg)
            module.update_qcfg(override_qcfg[name].get("input", None), override_qcfg[name]["output"])
    return model


def export_qcfg(model):
    output_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (QLinear, QRMSNorm, QLayerNorm, QMatMul, QSiLU, QGELU)):
            output_dict[name] = module.export_qcfg()
    return output_dict


def set_scale_and_offset(model, act_dict, use_scale_offset_as="buffer"):
    for name, module in model.named_modules():
        if isinstance(module, (QLinear, QRMSNorm, QLayerNorm, QMatMul, QSiLU, QGELU)):
            assert(name in act_dict)
            module.set_scale_offset(act_dict[name], use_scale_offset_as)
    return model
