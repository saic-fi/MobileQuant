import torch
import torch.nn as nn
import numpy as np
import os.path as osp
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
import os, argparse, gc, functools
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from mobilellm.utils.bench import print_model_size
from mobilellm.utils.io import json_load, json_save
from mobilellm.model.hf_model import HFForCausalLM, HFRMSNorm, HFDecoderLayer
from mobilellm.model.hf_config import HFConfig

AutoConfig.register("hfmodel", HFConfig)
AutoModelForCausalLM.register(HFConfig, HFForCausalLM)


parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', type=str, help='path of the hf model')
parser.add_argument('--act_scales_path', type=str, default=None, help='if we already pre-computed the act scales')
parser.add_argument('--alpha', type=float, default=0.5, help='smoothquant alpha')
parser.add_argument('--seq_len', type=int, default=4096, help='max seq len')
parser.add_argument('--calib_data', type=str, default='pileval', help='the calibration data')
parser.add_argument('--calib_path', type=str, default='data/pile/val.jsonl.zst', help='the calibration data')
parser.add_argument('--num_samples', type=int, default=512, help='num of calibration samples')
parser.add_argument('--use_rand_samples', default=False, action="store_true")
parser.add_argument('--original_smoothquant', default=False, action="store_true")
parser.add_argument('--original_omniquant', default=False, action="store_true")
parser.add_argument("--output_dir", default=None, type=str)
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.hf_path

if args.act_scales_path is None:
    args.act_scales_path = osp.join(args.hf_path, "act_scales.pth")


seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, (nn.LayerNorm, HFRMSNorm))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert len(ln.weight.data) == fc.in_features == len(act_scales)

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat([fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)).clamp(min=1e-5).to(device).to(dtype)

    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0

    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


# this is not included in the original smoothquant
# https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/smooth.py
@torch.no_grad()
def smooth_fc_fcs(fc1, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat([fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)).clamp(min=1e-5).to(device).to(dtype)
    
    fc1.weight.div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
    
    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0

    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5, original_smoothquant=False, original_omniquant=False):
    for name, module in tqdm(model.named_modules()):
        if isinstance(module, (HFDecoderLayer, )):
            if model.config.shared_attention_norm:
                attn_ln = module.input_layernorm
                fcs = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj, module.mlp.w1]
                if model.config.num_linears_per_mlp == 3:
                    fcs.append(module.mlp.w3)
                fcs_input_scales = scales[name + '.self_attn.q_proj_input']
                smooth_ln_fcs(attn_ln, fcs, fcs_input_scales, alpha)
            else:
                attn_ln = module.input_layernorm
                qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
                qkv_input_scales = scales[name + '.self_attn.q_proj_input']
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

                ffn_ln = module.post_attention_layernorm
                mlp = [module.mlp.w1]
                if model.config.num_linears_per_mlp == 3:
                    mlp.append(module.mlp.w3)
                mlp_input_scales = scales[name + '.mlp.w1_input']
                smooth_ln_fcs(ffn_ln, mlp, mlp_input_scales, alpha)

            # one of the main diffs from the originial smoothquant: https://github.com/mit-han-lab/smoothquant/blob/0cd1b7e8edc19747b1790ed5ca3db49281b31933/smoothquant/smooth.py#L75
            if not original_smoothquant:
                if module.self_attn.v_proj.weight.shape[0] == module.self_attn.o_proj.weight.shape[1]:
                    smooth_fc_fcs(module.self_attn.v_proj, module.self_attn.o_proj, scales[name + '.self_attn.o_proj_input'], alpha)
                # one of the main diffs from the original omniquant: https://github.com/OpenGVLab/OmniQuant/blob/8eca0ce9ae222e344e5ce0d8897ed9e2b3404b6e/quantize/utils.py#L77
                if not original_omniquant:
                    if model.config.num_linears_per_mlp == 3:
                        smooth_fc_fcs(module.mlp.w3, module.mlp.w2, scales[name + '.mlp.w2_input'], alpha)
                

@torch.no_grad()
def main():
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, use_fast=False, legacy=False)
    config = AutoConfig.from_pretrained(args.hf_path, trust_remote_code=True)
    config.use_matmul_as_module = True
    config._attn_implementation = "eager"
    config.l2norm_as_rmsnorm = True
    model = AutoModelForCausalLM.from_pretrained(args.hf_path, config=config, device_map='auto', torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="eager", use_safetensors=True)
    print(model)
    print_model_size(model, False)

    # calib data
    if args.calib_data == 'pileval':
        dataset = load_dataset("json", data_files=args.calib_path, split="train")
    elif args.calib_data == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split="train")
    else:
        dataset = load_dataset(args.calib_data, split="validation")

    dataset = dataset.shuffle(seed=seed)
    os.makedirs(args.output_dir, exist_ok=True)

    act_scales = torch.load(args.act_scales_path)
    
    smooth_lm(model, act_scales, args.alpha, args.original_smoothquant, args.original_omniquant)
    model.save_pretrained(args.output_dir, safe_serialization=False)
    

if __name__ == '__main__':
    main()