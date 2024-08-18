import torch
import torch.nn as nn
import os.path as osp
from tqdm import tqdm
import copy, math, gc, os, pdb
from contextlib import nullcontext
from collections import OrderedDict
from typing import List, Optional, Tuple, Union
from mobilellm.utils.parallel_utils import map_layers_to_multi_gpus
from mobilellm.utils.optim import NativeScalerWithGradNormCount
from mobilellm.model.hf_model import HFRMSNorm, HFDecoderLayer, FMatMul
from mobilellm.quantization.qmodule import QLinear, QLayerNorm, QRMSNorm, QMatMul, QSiLU, QGELU

try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")


CLIPMIN = 1e-5
CLIPMAX = 1e6

###############################################################################################
# Code adapted from https://github.com/OpenGVLab/OmniQuant/blob/main/models/transformation.py

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)


###################################################################################################
## Smooth
def smooth_ln_fcs_temporary(ln, fcs, scales, shifts):
    ln.use_temporary_parameter = True
    if not isinstance(fcs, list):
        fcs = [fcs]

    # device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    # scales = scales.to(device=device, dtype=dtype)
    # shifts = shifts.to(device=device, dtype=dtype)
    
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.temp_bias = (ln.bias - shifts) / scales
    else:
        ln.temp_bias = (-1 * shifts) / scales
    ln.temp_weight = ln.weight / scales

    for fc in fcs:
        fc.use_temporary_parameter = True
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.temp_bias = fc.bias + fc.weight@shifts
        else:
            fc.temp_bias = fc.weight@shifts
        fc.temp_weight = fc.weight * scales.view(1, -1)


def smooth_fc_fc_temporary(fc1, fc2, scales, shifts):
    fc1.use_temporary_parameter = True
    fc2.use_temporary_parameter = True

    if hasattr(fc1, 'temp_weight'):
        fc1.temp_bias = (fc1.temp_bias - shifts)/scales.view(-1)
        fc1.temp_weight = fc1.temp_weight/scales.view(-1, 1)
    else:
        #TODO: double check
        fc1.temp_bias = (fc1.bias - shifts)/scales.view(-1)
        fc1.temp_weight = fc1.weight/scales.view(-1, 1)
    
    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.temp_bias = fc2.bias + fc2.weight @ shifts
    else:
        fc2.temp_bias = fc2.weight @ shifts
    fc2.temp_weight = fc2.weight * scales.view(1, -1)


def smooth_q_k_temporary(q_proj, k_proj, scales):
    q_proj.use_temporary_parameter = True
    k_proj.use_temporary_parameter = True
    q_proj.temp_weight = q_proj.temp_weight/scales.view(-1,1)
    q_proj.temp_bias = q_proj.temp_bias/scales.view(-1)
    k_proj.temp_weight = k_proj.temp_weight*scales.view(-1,1)
    k_proj.temp_bias = k_proj.temp_bias*scales.view(-1)


def smooth_ln_fcs_inplace(ln, fcs, scales, shifts):
    ln.use_temporary_parameter = False

    if not isinstance(fcs, list):
        fcs = [fcs]
    
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.sub_(shifts)
        ln.bias.div_(scales)
    else:
        if hasattr(ln, "bias"): del ln.bias
        ln.register_buffer('bias', (-1 * shifts)/scales)

    ln.weight.div_(scales)
    for fc in fcs:
        fc.use_temporary_parameter = False
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.bias.add_(fc.weight @ shifts)
        else:
            if hasattr(fc, 'bias'): del fc.bias
            fc.register_buffer('bias', fc.weight @ shifts)
        fc.weight.mul_(scales.view(1, -1))


def smooth_fc_fc_inplace(fc1, fc2, scales, shifts):
    fc1.use_temporary_parameter = False
    fc2.use_temporary_parameter = False
    fc1.bias.sub_(shifts)
    fc1.bias.div_(scales.view(-1))
    fc1.weight.div_(scales.view(-1,1))
    
    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.bias.add_(fc2.weight @ shifts)
    else:
        del fc2.bias
        fc2.register_buffer('bias', fc2.weight @ shifts)
    fc2.weight.mul_(scales.view(1, -1))


def smooth_q_k_inplace(q_proj, k_proj, scales):
    q_proj.use_temporary_parameter = False
    k_proj.use_temporary_parameter = False
    q_proj.weight.div_(scales.view(-1, 1))
    q_proj.bias.div_(scales.view(-1))
    k_proj.weight.mul_(scales.view(-1, 1))
    k_proj.bias.mul_(scales.view(-1))


@torch.no_grad()   
def smooth_lm_inplace(model, config, use_let, use_shift = False, original_omniquant = False):
    if use_let:
        template = "smooth" if use_shift else "smooth_scale"
        for name, module in model.named_parameters():
            if template in name:
                module.data = truncate_number(module)

        if config.shared_attention_norm:
            attn_ln = model.input_layernorm
            fcs = [model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj, model.mlp.w1]
            if config.num_linears_per_mlp == 3:
                fcs.append(model.mlp.w3)
            smooth_ln_fcs_inplace(attn_ln, fcs, model.qkv_smooth_scale, model.qkv_smooth_shift)
        else:
            attn_ln = model.input_layernorm
            fcs = [model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj]
            smooth_ln_fcs_inplace(attn_ln, fcs, model.qkv_smooth_scale, model.qkv_smooth_shift)

            ffn_ln = model.post_attention_layernorm
            mlp = [model.mlp.w1]
            if config.num_linears_per_mlp == 3:
                mlp.append(model.mlp.w3)
            smooth_ln_fcs_inplace(ffn_ln, mlp, model.fc1_smooth_scale, model.fc1_smooth_shift)

        if model.self_attn.v_proj.weight.shape[0] == model.self_attn.o_proj.weight.shape[1]:
            smooth_fc_fc_inplace(model.self_attn.v_proj, model.self_attn.o_proj, model.out_smooth_scale, model.out_smooth_shift)
        
        if config.num_linears_per_mlp == 3 and not original_omniquant:
            smooth_fc_fc_inplace(model.mlp.w3, model.mlp.w2, model.fc2_smooth_scale, model.fc2_smooth_shift)

        if model.self_attn.q_proj.weight.shape[0] == model.self_attn.k_proj.weight.shape[0]:
            smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj, model.qkt_smooth_scale)
    
    for name, module in model.named_modules():
        if isinstance(module, (QLinear, QRMSNorm, QLayerNorm)):
            module.weight.data = module.weight_quantizer.run_lwc(module.weight)
            module.use_temporary_parameter = False


def smooth_lm_temporary(model, config, use_let, use_shift = False, original_omniquant = False):
    if use_let:
        template = "smooth" if use_shift else "smooth_scale"
        with torch.no_grad():
            for name, module in model.named_parameters():
                if template in name:
                    module.data = truncate_number(module)

        if config.shared_attention_norm:
            attn_ln = model.input_layernorm
            fcs = [model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj, model.mlp.w1]
            if config.num_linears_per_mlp == 3:
                fcs.append(model.mlp.w3)
            smooth_ln_fcs_temporary(attn_ln, fcs, model.qkv_smooth_scale, model.qkv_smooth_shift)
        else:
            attn_ln = model.input_layernorm
            fcs = [model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj]
            smooth_ln_fcs_temporary(attn_ln, fcs, model.qkv_smooth_scale, model.qkv_smooth_shift)

            ffn_ln = model.post_attention_layernorm
            mlp = [model.mlp.w1]
            if config.num_linears_per_mlp == 3:
                mlp.append(model.mlp.w3)
            smooth_ln_fcs_temporary(ffn_ln, mlp, model.fc1_smooth_scale, model.fc1_smooth_shift)
        
        if model.self_attn.v_proj.weight.shape[0] == model.self_attn.o_proj.weight.shape[1]:
            smooth_fc_fc_temporary(model.self_attn.v_proj, model.self_attn.o_proj, model.out_smooth_scale, model.out_smooth_shift)

        if config.num_linears_per_mlp == 3 and not original_omniquant:
            # one of the main diffs from the original omniquant: https://github.com/OpenGVLab/OmniQuant/blob/8eca0ce9ae222e344e5ce0d8897ed9e2b3404b6e/quantize/utils.py#L77
            smooth_fc_fc_temporary(model.mlp.w3, model.mlp.w2, model.fc2_smooth_scale, model.fc2_smooth_shift)

        if model.self_attn.q_proj.weight.shape[0] == model.self_attn.k_proj.weight.shape[0]:
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj, model.qkt_smooth_scale)
    else:
        for name, module in model.named_modules():
            if isinstance(module, QLinear):
                module.temp_weight = module.weight
                module.temp_bias = module.bias
    
    for name, module in model.named_modules():
        if isinstance(module, QLinear):
            module.use_temporary_parameter = True
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            if not hasattr(module, "temp_weight"):
                module.temp_weight = module.weight
###################################################################################################


###################################################################################################
## Params
def let_parameters(model, use_shift=False):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for name, param in model.named_parameters():
        if name.find(template) > -1:
            params.append(param)
    return iter(params)  


def lwc_parameters(model):
    params = []
    for name, param in model.named_parameters():
        if name.find('bound_factor') > -1:
            params.append(param)
    return iter(params)  


def lrl_parameters(model):
    params = []
    for name, param in model.named_parameters():
        if name.find('quantizer.offset') > -1:
            params.append(param)
        if name.find('quantizer.scale') > -1:
            params.append(param)
    return iter(params)


def get_parameters(model, use_shift=False):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for name, param in model.named_parameters():
        if name.find('bound_factor') > -1 or name.find(template) > -1 or name.find('quantizer.offset') > -1 or name.find('quantizer.scale') > -1:
            params.append(param)
    return iter(params)


def quant_state_dict(model, destination=None, prefix='', keep_vars=False, use_shift=False):
    if destination is None:
        destination = OrderedDict()
    template = "smooth" if use_shift else "smooth_scale"
    for name, param in model.named_parameters():
        if name.find(template) > -1 or name.find('bound_factor') > -1 or name.find('quantizer.offset') > -1 or name.find('quantizer.scale') > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination


def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, (QLinear, QLayerNorm, QRMSNorm)):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias
###################################################################################################


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(max_lr: float, min_lr: float, it: int, warmup_iters: int, max_iters: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)


###################################################################################################
## Quant alg.

class LayerList(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None):
        for i in range(len(self.layers)):
            out = self.layers[i](hidden_states, attention_mask, position_ids)
            hidden_states = out[0]
        return out


def enable_quant(args, model):
    for name, module in reversed(model._modules.items()):
        if isinstance(module, (QLinear, )):
            if model._modules[name].input_quantizer is not None:
                model._modules[name].input_quantizer.enable  = True
            model._modules[name].weight_quantizer.enable = True
            if args.lwc:
                model._modules[name].weight_quantizer.enable_lwc(model._modules[name].weight)
            model._modules[name].output_quantizer.enable = True
        elif isinstance(module, (QRMSNorm, QLayerNorm)):
            model._modules[name].input_quantizer.enable  = True
            model._modules[name].weight_quantizer.enable = True
            if args.lwc:
                model._modules[name].weight_quantizer.enable_lwc(model._modules[name].weight)
            model._modules[name].output_quantizer.enable = True
        elif isinstance(module, (QMatMul, QSiLU)):
            if model._modules[name].input_quantizer is not None:
                model._modules[name].input_quantizer.enable  = True
            model._modules[name].input2_quantizer.enable = True
            model._modules[name].output_quantizer.enable = True
        elif isinstance(module, (QGELU, )):
            if model._modules[name].input_quantizer is not None:
                model._modules[name].input_quantizer.enable  = True
            model._modules[name].output_quantizer.enable = True
        elif len(list(module.children())) > 1:
            enable_quant(args, module)
    return model


def disable_quant(model):
    for name, module in reversed(model._modules.items()):
        if isinstance(module, (QLinear, )):
            if model._modules[name].input_quantizer is not None:
                model._modules[name].input_quantizer.enable  = False
            model._modules[name].weight_quantizer.enable = False
            model._modules[name].weight_quantizer.disable_lwc()
            model._modules[name].output_quantizer.enable = False
        elif isinstance(module, (QRMSNorm, QLayerNorm)):
            model._modules[name].input_quantizer.enable  = False
            model._modules[name].weight_quantizer.enable = False
            model._modules[name].weight_quantizer.disable_lwc()
            model._modules[name].output_quantizer.enable = False
        elif isinstance(module, (QMatMul, QSiLU)):
            if model._modules[name].input_quantizer is not None:
                model._modules[name].input_quantizer.enable  = False
            model._modules[name].input2_quantizer.enable = False
            model._modules[name].output_quantizer.enable = False
        elif isinstance(module, (QGELU, )):
            if model._modules[name].input_quantizer is not None:
                model._modules[name].input_quantizer.enable  = False
            model._modules[name].output_quantizer.enable = False
        elif len(list(module.children())) > 1:
            disable_quant(module)
    return model


def omniquant(args, model, dataloader, logger, device=None):
    logger.info("Starting ...")
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if device is None:
        device = next(model.parameters()).device
    
    # move embedding layer and the norm layer to target device
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)

    layer_name_prefix = "model.layers"
    pairs = {"q_proj": "qkv", "w1": "fc1" }
    if layers[0].self_attn.v_proj.weight.shape[0] == layers[0].self_attn.o_proj.weight.shape[1]:
        pairs["o_proj"] = "out"
    if model.config.num_linears_per_mlp == 3 and not args.original_omniquant:
        pairs["w2"] = "fc2"
    
    # move the first layer to the target device
    layers[0] = layers[0].to(device)
    if args.deactive_amp and args.epochs > 0:
        dtype = torch.float32
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros((args.nsamples, args.seqlen, model.config.hidden_size), dtype=dtype, device=device)

    # cache the first layer input in "inps"
    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(device))
            except ValueError:
                pass

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    if not args.cache_in_gpu:
        inps = inps.cpu()
    quant_inps  = inps
    fp_inps     = copy.deepcopy(inps) # take output of fp model as input
    fp_inps_2   = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size, 1, 1, 1) if args.deactive_amp else attention_mask.repeat(args.batch_size, 1, 1, 1).float()
    else:
        logger.info("No attention mask caught from the first layer. Seems that model's attention works without a mask.")
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}
    
    ###########################################################################################################
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        qlayer = layers[i].to(device)
        ########################################################################################################
        # obtain output of full-precision model
        disable_quant(qlayer)
        if args.epochs > 0:
            with torch.no_grad():
                # TODO: full-precision context
                with traincast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0).to(device), attention_mask=attention_mask, position_ids=position_ids)[0].to(fp_inps.device)
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0).to(device), attention_mask=attention_mask, position_ids=position_ids)[0].to(fp_inps_2.device)
        
        # init smooth parameters
        enable_quant(args, qlayer)

        if args.let:
            # init channel-wise scaling and shift
            # new_parameters
            if qlayer.self_attn.q_proj.weight.shape[0] == qlayer.self_attn.k_proj.weight.shape[0]:
                # for self-attn
                qlayer.register_parameter("qkt_smooth_scale", torch.nn.Parameter(torch.ones(qlayer.self_attn.q_proj.out_features, device=device, dtype=dtype)))

            for name, module in qlayer.named_modules():
                if isinstance(module, QLinear):
                    for key in pairs.keys():
                        if key in name:
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift", torch.nn.Parameter(torch.zeros(module.in_features, device=device, dtype=dtype)))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale", torch.nn.Parameter(torch.ones(module.in_features, device=device, dtype=dtype)))
                                
        if args.resume:
            msg = qlayer.load_state_dict(omni_parameters[i], strict=False)
            print(msg)
        
        if args.epochs > 0:
            if args.dtype != "float32" and not args.deactive_amp:
                with torch.no_grad():
                    qlayer.float()      # required for AMP training
            # create optimizer
            trainable_params = [
                {"params":  let_parameters(qlayer, args.use_shift), "lr": args.let_lr}, 
                {"params":  lwc_parameters(qlayer), "lr": args.lwc_lr},
            ]
            if args.lrl:
                trainable_params.append({"params":  lrl_parameters(qlayer), "lr": args.lrl_lr})
            optimizer = torch.optim.AdamW(trainable_params,  weight_decay=args.wd)
            loss_scaler = NativeScalerWithGradNormCount()
            
            max_iters = args.epochs * (args.nsamples // args.batch_size)
            warmup_iters = args.warmup_epochs * (args.nsamples // args.batch_size)
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                completed_steps = epochs * (args.nsamples // args.batch_size)
                for j in range(args.nsamples // args.batch_size):    
                    index = j * args.batch_size
                    optimizer.param_groups[0]['lr'] = get_lr(args.let_lr, args.let_min_lr, completed_steps + j, warmup_iters, max_iters)
                    optimizer.param_groups[1]['lr'] = get_lr(args.lwc_lr, args.lwc_min_lr, completed_steps + j, warmup_iters, max_iters)
                    if args.lrl:
                        optimizer.param_groups[2]['lr'] = get_lr(args.lrl_lr, args.lrl_min_lr, completed_steps + j, warmup_iters, max_iters)
                    # obtain output of quantization model
                    with traincast():
                        smooth_lm_temporary(qlayer, model.config, args.let, args.use_shift, args.original_omniquant)
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,].to(device), attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        loss = loss_func(fp_inps[index:index+args.batch_size,].to(device), quant_out)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,].to(device), quant_out)
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer, parameters=get_parameters(qlayer, args.use_shift)).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(device) / 1024**2} ")
            clear_temp_variable(qlayer)
            del optimizer
        
        qlayer.to(args.dtype)

        # real smooth and quantization
        if args.epochs > 0:
            omni_parameters[i] = quant_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"quant_parameters.pth"))
        smooth_lm_inplace(qlayer, model.config, args.let, args.use_shift, args.original_omniquant)

        # remove quant parameters
        params_to_del = []
        for name, param in qlayer.named_parameters():
            if name.find('bound_factor') > -1 or name.find("smooth") > -1:
                params_to_del.append(name)
        for name in params_to_del:
            delattr(qlayer, name)

        if args.epochs > 0:
            # update input of quantization model
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0).to(device), attention_mask=attention_mask,position_ids=position_ids)[0].cpu()
        layers[i] = qlayer.to("cpu")
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model


def e2equant(args, model, dataloader, logger, device=None):
    logger.info("Starting ...")
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    map_layers_to_multi_gpus(layers)

    input_device = model.model.layers[0].device
    input_dtype = model.model.layers[0].self_attn.v_proj.weight.dtype
    output_device = model.model.layers[-1].device
    assert input_device == output_device
    model.model.embed_tokens.to(input_device)
    model.model.norm.to(output_device)
    model.lm_head.to(output_device)


    pairs = {"q_proj": "qkv", "w1": "fc1" }
    if layers[0].self_attn.v_proj.weight.shape[0] == layers[0].self_attn.o_proj.weight.shape[1]:
        pairs["o_proj"] = "out"
    if model.config.num_linears_per_mlp == 3:
        pairs["w2"] = "fc2"

    if args.deactive_amp and args.epochs > 0:
        dtype = torch.float32
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast

    inps = torch.zeros((args.nsamples, args.seqlen, model.config.hidden_size), dtype=input_dtype, device=input_device)
    # cache the first layer input in "inps"
    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(input_device))
            except ValueError:
                pass

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    # same input of first layer for fp model and quant model
    if not args.cache_in_gpu:
        inps = inps.cpu()
    quant_inps  = inps
    fp_inps     = copy.deepcopy(inps) # take output of fp model as input
    fp_inps_2   = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size, 1, 1, 1) if args.deactive_amp else attention_mask.repeat(args.batch_size, 1, 1, 1).float()
    else:
        logger.info("No attention mask caught from the first layer. Seems that model's attention works without a mask.")
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if args.resume:
        e2e_parameters = torch.load(args.resume)
    else:
        e2e_parameters = {}


    batch_size = args.batch_size
    disable_quant(model)
    backbone = LayerList(layers)

    if args.epochs > 0:
        with torch.no_grad():
            with traincast():
                for j in tqdm(range(args.nsamples // args.batch_size)):    
                    index = j * args.batch_size
                    fp_inps[index:(index+batch_size)] = backbone(
                        fp_inps[index:(index+batch_size)], 
                        attention_mask=attention_mask_batch, 
                        position_ids=position_ids
                    )[0].to(fp_inps.device)
                    if args.aug_loss:
                        fp_inps_2[index:(index+batch_size)] = backbone(
                            quant_inps[index:(index+batch_size)], 
                            attention_mask=attention_mask_batch, 
                            position_ids=position_ids
                        )[0].to(fp_inps_2.device)
    torch.cuda.empty_cache()
    enable_quant(args, model)

    if args.let:
        for i in range(len(layers)):
            device = layers[i].self_attn.q_proj.weight.device 
            dtype = layers[i].self_attn.q_proj.weight.dtype
            if layers[i].self_attn.q_proj.weight.shape[0] == layers[i].self_attn.k_proj.weight.shape[0]:
                layers[i].register_parameter(
                    "qkt_smooth_scale", 
                    torch.nn.Parameter(torch.ones(layers[i].self_attn.q_proj.out_features, device=device, dtype=dtype))
                )
            for name, module in layers[i].named_modules():
                if isinstance(module, QLinear):
                    for key in pairs.keys():
                        if key in name:
                            layers[i].register_parameter(f"{pairs[key]}_smooth_shift", torch.nn.Parameter(torch.zeros(module.in_features, device=device, dtype=dtype)))
                            layers[i].register_parameter(f"{pairs[key]}_smooth_scale", torch.nn.Parameter(torch.ones(module.in_features, device=device, dtype=dtype)))
            if args.resume:
                msg = layers[i].load_state_dict(e2e_parameters[i], strict=True)
                print(msg)

    if args.epochs > 0:
        if args.dtype != "float32" and not args.deactive_amp:
            with torch.no_grad():
                model = model.to(torch.float32)      # required for AMP training
        # create optimizer
        optimizer = torch.optim.AdamW(
            [
                {"params":  let_parameters(model, args.use_shift), "lr": args.let_lr}, 
                {"params":  lwc_parameters(model), "lr": args.lwc_lr},
                {"params":  lrl_parameters(model), "lr": args.lrl_lr}
            ],  weight_decay=args.wd
        )
        loss_scaler = NativeScalerWithGradNormCount()

        max_iters = args.epochs * (args.nsamples // args.batch_size)
        warmup_iters = args.warmup_epochs * (args.nsamples // args.batch_size)

        
        for epochs in range(args.epochs):
            loss_list = []
            norm_list = []
            completed_steps = epochs * (args.nsamples // args.batch_size)
            for j in tqdm(range(args.nsamples // args.batch_size)):    
                index = j * args.batch_size
                
                optimizer.param_groups[0]['lr'] = get_lr(args.let_lr, args.let_min_lr, completed_steps + j, warmup_iters, max_iters)
                optimizer.param_groups[1]['lr'] = get_lr(args.lwc_lr, args.lwc_min_lr, completed_steps + j, warmup_iters, max_iters)
                optimizer.param_groups[2]['lr'] = get_lr(args.lrl_lr, args.lrl_min_lr, completed_steps + j, warmup_iters, max_iters)

                # obtain output of quantization model
                with traincast():
                    for k in range(len(layers)):
                        smooth_lm_temporary(layers[k], model.config, args.let, args.use_shift)
                    quant_out = backbone(quant_inps[index:index+args.batch_size,].to(input_device), attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                    loss = loss_func(fp_inps[index:index+args.batch_size].to(output_device), quant_out)
                    if args.aug_loss:
                        loss += loss_func(fp_inps_2[index:index+args.batch_size,].to(output_device), quant_out)
                if not math.isfinite(loss.item()):
                    logger.info("Loss is NAN, stopping training")
                    pdb.set_trace()
                    
                loss_list.append(loss.detach().cpu())
                optimizer.zero_grad()
                norm = loss_scaler(loss, optimizer, parameters=get_parameters(model, args.use_shift)).cpu()
                norm_list.append(norm.data)

            loss_mean = torch.stack(loss_list).mean()
            norm_mean = torch.stack(norm_list).mean()
            logger.info(f"Epoch {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(device) / 1024**2} ")

            for k in range(len(layers)):
                e2e_parameters[k] = quant_state_dict(layers[k])
            torch.save(e2e_parameters, os.path.join(args.output_dir, f"parameters.pth"))
    
    # clear_temp_variable(layers)
    for i in range(len(layers)):
        e2e_parameters[i] = quant_state_dict(layers[i])
        smooth_lm_inplace(layers[i], model.config, args.let, args.use_shift)
        # remove omniquant parameters
        params_to_del = []
        for name, param in layers[i].named_parameters():
            if name.find('bound_factor') > -1 or name.find("smooth") > -1:
                params_to_del.append(name)
        for name in params_to_del:
            delattr(layers[i], name)
        
    torch.save(e2e_parameters, os.path.join(args.output_dir, f"parameters.pth"))

    del optimizer
    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model