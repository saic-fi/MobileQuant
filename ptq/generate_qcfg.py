import torch
import torch.nn as nn
import numpy as np
import os.path as osp
from tqdm import tqdm
from copy import deepcopy
from functools import partial
import os, argparse, gc, functools
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from mobilellm.utils.bench import print_model_size
from mobilellm.utils.io import json_load, json_save
from mobilellm.model.hf_model import HFForCausalLM
from mobilellm.model.hf_config import HFConfig
from mobilellm.quantization.qmodule import QuantConfig, Quantizer, QLinear, QRMSNorm, QLayerNorm, QMatMul, QSiLU, QGELU, set_scale_and_offset, update_qcfg, export_qcfg, create_sim_qmodel


AutoConfig.register("hfmodel", HFConfig)
AutoModelForCausalLM.register(HFConfig, HFForCausalLM)


parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', type=str, help='path of the hf model')
parser.add_argument('--weight_bitwidth', type=int, default=8, help='default bitwidth for the weight')
parser.add_argument('--weight_group_size', type=int, default=-1, help='group size if we like to use per-group quantization')
parser.add_argument('--weight_is_per_channel', default=False, action="store_true", help=' if we like to use per-channel quantization')
parser.add_argument('--weight_is_symmetric', default=False, action="store_true", help=' if we like to use symmetric quantization')
parser.add_argument('--act_bitwidth', type=int, default=8, help='default bitwidth for the activation')
parser.add_argument('--act_group_size', type=int, default=-1, help='group size if we like to use per-group quantization')
parser.add_argument('--act_is_per_channel', default=False, action="store_true", help=' if we like to use per-channel quantization')
parser.add_argument('--act_is_symmetric', default=False, action="store_true", help=' if we like to use symmetric quantization')
parser.add_argument('--act_is_dynamic', default=False, action="store_true", help=' if we like to use dynamic quantization')
parser.add_argument('--use_16bit_output_for_mlp', default=False, action="store_true", help='for gemma')
parser.add_argument('--use_16bit_softmax_input', default=False, action="store_true", help='for phi')
parser.add_argument('--use_16bit_softmax_output', default=False, action="store_true", help='for phi')
parser.add_argument("--output_dir", default=None, type=str)
args = parser.parse_args()


if args.output_dir is None:
    args.output_dir = args.hf_path

if args.weight_group_size != -1:
    assert args.weight_is_per_channel, "weight_is_per_channel should be activated if we'd like to use per-group quantization (i.e. weight_group_size != -1)"


seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


@torch.no_grad()
def main():
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, use_fast=False, legacy=False)
    config = AutoConfig.from_pretrained(args.hf_path, trust_remote_code=True)
    config.use_matmul_as_module = True
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.hf_path, config=config, device_map='auto', low_cpu_mem_usage=True, trust_remote_code=True)
    print(model)
    print_model_size(model, False)

    # qcfg 
    # default weight qcfg
    default_weight_qcfg = QuantConfig()
    default_weight_qcfg.bitwidth = args.weight_bitwidth
    default_weight_qcfg.group_size = args.weight_group_size
    default_weight_qcfg.is_symmetric = args.weight_is_symmetric
    default_weight_qcfg.is_per_channel = args.weight_is_per_channel
    
    # default act qcfg
    default_act_qcfg = QuantConfig()
    default_act_qcfg.bitwidth = args.act_bitwidth
    default_act_qcfg.group_size = args.act_group_size
    default_act_qcfg.is_symmetric = args.act_is_symmetric
    default_act_qcfg.is_dynamic = args.act_is_dynamic
    default_act_qcfg.is_per_channel = args.act_is_per_channel

    model = create_sim_qmodel(model, default_weight_qcfg, default_act_qcfg)

    #####################################################################
    ## mixed-precision
    def create_mixed_precision_model(model):
        for name, module in reversed(model._modules.items()):
            if isinstance(module, QLinear):
                if 'w2' in name:
                    model._modules[name].weight_quantizer.qcfg.is_per_channel = True
                    model._modules[name].output_quantizer.qcfg.bitwidth = 16
                elif 'o_proj' in name:
                    model._modules[name].output_quantizer.qcfg.bitwidth = 16
                if args.use_16bit_output_for_mlp and ('w1' in name or 'w3' in name):
                    model._modules[name].output_quantizer.qcfg.bitwidth = 16
            elif isinstance(module, (QRMSNorm, QLayerNorm)):
                model._modules[name].input_quantizer.qcfg.bitwidth = 16
                model._modules[name].weight_quantizer.qcfg.bitwidth = 16
            elif isinstance(module, QMatMul):
                if 'qk_bmm' in name and args.use_16bit_softmax_input:
                    model._modules[name].output_quantizer.qcfg.bitwidth = 16
                if 'pv_bmm' in name and args.use_16bit_softmax_output:
                    model._modules[name].input_quantizer.qcfg.bitwidth = 16
            elif isinstance(module, QSiLU):
                # the input has been quantized by w1
                if model._modules[name].input_quantizer is not None:
                    model._modules[name].input_quantizer.enable = False
            elif isinstance(module, QGELU):
                # the input has been quantized by w1
                if model._modules[name].input_quantizer is not None:
                    model._modules[name].input_quantizer.enable = False
            elif len(list(module.children())) > 1:
                create_mixed_precision_model(module)
        return model
    
    model = create_mixed_precision_model(model)
    os.makedirs(args.output_dir, exist_ok=True)
    default_qcfg = export_qcfg(model)
    json_save(osp.join(args.output_dir, 'default_qcfg.json'), default_qcfg)

    # if args.override_qcfg_path is not None:
    #     override_qcfg = json_load(args.override_qcfg_path)
    #     model = update_qcfg(model, override_qcfg)
    #     current_qcfg = export_qcfg(model)
    #     json_save(osp.join(args.output_dir, 'current_qcfg.json'), current_qcfg)
    # else:
    #     default_qcfg = export_qcfg(model)
    #     json_save(osp.join(args.output_dir, 'default_qcfg.json'), default_qcfg)

    # if args.act_dict_path is not None:
    #     # pre-computed activation range
    #     act_dict = json_load(args.act_dict_path)
    #     model = set_scale_and_offset(model, act_dict, 'parameter')
    # print_model_size(model, True)

    
if __name__ == '__main__':
    main()