import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from lm_eval import evaluator, tasks, utils
import os, sys, random, pdb, time, argparse

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from mobilellm.data.datautils import get_loaders
from mobilellm.model.hf_model import HFForCausalLM, HFDecoderLayer
from mobilellm.model.hf_config import HFConfig
from mobilellm.utils.bench import print_model_size, LMEvalAdaptor
from mobilellm.utils.io import json_load, json_save, create_logger
from mobilellm.quantization.algorithm import omniquant, e2equant
from mobilellm.quantization.qmodule import QuantConfig, QLinear, QRMSNorm, QLayerNorm, QMatMul, QSiLU, QGELU
from mobilellm.quantization.qmodule import create_fp_model, export_act_range, create_sim_qmodel, create_weight_only_qmodel, set_scale_and_offset, update_qcfg, export_qcfg


AutoConfig.register("hfmodel", HFConfig)
AutoModelForCausalLM.register(HFConfig, HFForCausalLM)


parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', type=str, default=None, help='path of the hf model')
parser.add_argument('--dtype', type=str, default=None, help='dtype of the hf model')
parser.add_argument("--output_dir", default="results/quant", type=str, help="direction of logging file")
parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--calib_dataset", type=str, default="pile", choices=["wikitext2","pile"], help="calibration set.")
parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
parser.add_argument('--seqlen', type=int, default=2048, help='max seq len')
parser.add_argument('--act_dict_path', type=str, default=None, help='the act dict file for custom quantization')
parser.add_argument('--override_qcfg_path', type=str, default=None, help='the fine-grained config for custom quantization')

parser.add_argument('--weight_bitwidth', type=int, default=4, help='default bitwidth for the weight')
parser.add_argument('--weight_group_size', type=int, default=-1, help='group size if we like to use per-group quantization')
parser.add_argument('--weight_is_per_channel', default=False, action="store_true", help=' if we like to use per-channel quantization')
parser.add_argument('--weight_is_symmetric', default=False, action="store_true", help=' if we like to use symmetric quantization')
parser.add_argument('--weight_is_dynamic', default=False, action="store_true", help=' if we like to use dynamic quantization')

parser.add_argument('--act_bitwidth', type=int, default=16, help='default bitwidth for the activation')
parser.add_argument('--act_group_size', type=int, default=-1, help='group size if we like to use per-group quantization')
parser.add_argument('--act_is_per_channel', default=False, action="store_true", help=' if we like to use per-channel quantization')
parser.add_argument('--act_is_symmetric', default=False, action="store_true", help=' if we like to use symmetric quantization')
parser.add_argument('--act_is_dynamic', default=False, action="store_true", help=' if we like to use dynamic quantization')

parser.add_argument("--let",    default=False, action="store_true", help="activate learnable equivalent transformation")
parser.add_argument("--lwc",    default=False, action="store_true", help="activate learnable weight clipping")
parser.add_argument("--lrl",    default=False, action="store_true", help="activate learnable activation range")
parser.add_argument("--let_lr", type=float, default=1e-3)
parser.add_argument("--lwc_lr", type=float, default=1e-2)
parser.add_argument("--lrl_lr", type=float, default=1e-6)
parser.add_argument("--let_min_lr", type=float, default=1e-3)
parser.add_argument("--lwc_min_lr", type=float, default=1e-2)
parser.add_argument("--lrl_min_lr", type=float, default=1e-6)
parser.add_argument("--wd",     type=float, default=0)
parser.add_argument("--epochs", type=int,   default=10)
parser.add_argument("--warmup_epochs", type=int,   default=0)
parser.add_argument("--use_shift",  default=False, action="store_true", help="learn the shift")
parser.add_argument("--aug_loss",   default=False, action="store_true", help="calculate additional loss with same input")
parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_fewshot', type=int, default=0, help='number of fewshot examples')
parser.add_argument("--tasks", default="wikitext", type=str, help='llm task from the harness repo')
parser.add_argument("--mode", default="omniquant",  type=str, choices=["e2e", "omniquant"])
parser.add_argument("--original_omniquant", default=False, action="store_true")
parser.add_argument("--cache_in_gpu", default=False, action="store_true")
parser.add_argument('--use_8bit_softmax_input', default=False, action="store_true", help='for phi')
parser.add_argument('--use_8bit_softmax_output', default=False, action="store_true", help='for phi')
args = parser.parse_args()


args.model_family = osp.basename(args.hf_path).split('-')[0]

if args.act_dict_path is None:
    args.act_dict_path = osp.join(args.hf_path, "act_dict.json")

if args.override_qcfg_path is None:
    args.override_qcfg_path = osp.join(args.hf_path, "default_qcfg.json")


seed = 1337
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


STR_TO_DTYPE = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


@torch.no_grad()
def evaluate(args, model, tokenizer, logger):
    results = {}
    input_device  = model.model.embed_tokens.weight.device
    output_device = model.lm_head.weight.device
    task_names = args.tasks.split(",")
    task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
    print(f"Running tasks: {task_names}")
    model_name = osp.basename(args.hf_path)
    lm_eval_model = LMEvalAdaptor(model_name, model, tokenizer, args.batch_size, max_length=args.seqlen)
    t_results = evaluator.simple_evaluate(model=lm_eval_model, tasks=task_names, batch_size=args.batch_size, no_cache=True, num_fewshot=args.num_fewshot)
    results.update(t_results)
    logger.info(results)
    pprint(results)
    results["config"]["model"] = model_name
    os.makedirs(args.output_dir, exist_ok=True)
    json_save(osp.join(args.output_dir, 'results.json'), results)
    return results


def main():
    if args.epochs > 0:
        assert args.lwc or args.let or args.lrl

    if (args.weight_bitwidth < 16 and args.weight_bitwidth >= 8) or (args.act_bitwidth < 16 and args.act_bitwidth >= 8):
        args.deactive_amp = True

    # output dir
    if args.output_dir:
        Path(args.output_dir).mkdir(parents = True, exist_ok = True)
    output_dir = Path(args.output_dir)

    # logger
    logger = create_logger(output_dir)
    logger.info(args)

    # cache dir
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    
    # load model and tokenizer
    config = AutoConfig.from_pretrained(args.hf_path, trust_remote_code=True)
    config.use_matmul_as_module = True
    config._attn_implementation = "eager"
    config.l2norm_as_rmsnorm = True

    if args.dtype is None:
        args.dtype = config.torch_dtype
    else:
        args.dtype = STR_TO_DTYPE[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, use_fast=False, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(args.hf_path, config=config, device_map='cpu', torch_dtype=args.dtype, low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="eager",
        use_safetensors=False, #args.act_bitwidth > 16 # TODO
    ) 

    weight_qcfg, act_qcfg = QuantConfig(), QuantConfig()

    weight_qcfg.bitwidth        = args.weight_bitwidth
    weight_qcfg.group_size      = args.weight_group_size
    weight_qcfg.is_symmetric    = args.weight_is_symmetric
    weight_qcfg.is_per_channel  = args.weight_is_per_channel
    weight_qcfg.is_dynamic      = args.weight_is_dynamic

    act_qcfg.bitwidth        = args.act_bitwidth
    act_qcfg.group_size      = args.act_group_size
    act_qcfg.is_symmetric    = args.act_is_symmetric
    act_qcfg.is_per_channel  = args.act_is_per_channel
    act_qcfg.is_dynamic      = args.act_is_dynamic

    #############################################################################
    # simulation model
    model = create_sim_qmodel(model, weight_qcfg, act_qcfg)
    # model.eval()
    for param in model.parameters():
        param.requires_grad = False

    def update_quant_cfg(model):
        for name, module in reversed(model._modules.items()):
            if isinstance(module, QLinear):
                if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name or "w1" in name or "w3" in name:
                    model._modules[name].input_quantizer = None
                if 'w2' in name:
                    model._modules[name].weight_quantizer.qcfg.is_per_channel = True
                    model._modules[name].output_quantizer.qcfg.bitwidth = 16
                elif 'o_proj' in name:
                    model._modules[name].output_quantizer.qcfg.bitwidth = 16
            elif isinstance(module, (QRMSNorm, QLayerNorm)):
                model._modules[name].input_quantizer.qcfg.bitwidth = 16
                model._modules[name].weight_quantizer.qcfg.bitwidth = 16
                model._modules[name].weight_quantizer.qcfg.is_symmetric = False
                model._modules[name].weight_quantizer.qcfg.is_per_channel = False
            elif isinstance(module, QMatMul):
                if 'qk_bmm' in name and (not args.use_8bit_softmax_input):
                    model._modules[name].output_quantizer.qcfg.bitwidth = 16
                if 'pv_bmm' in name and (not args.use_8bit_softmax_output):
                    model._modules[name].input_quantizer.qcfg.bitwidth = 16
            elif isinstance(module, QSiLU):
                model._modules[name].input_quantizer = None
            elif isinstance(module, QGELU):
                model._modules[name].input_quantizer = None
            elif len(list(module.children())) > 1:
                update_quant_cfg(module)
        return model
    
    model = update_quant_cfg(model)
    # pre-computed activation range
    if not args.act_is_dynamic:
        # static activation ranges
        if args.act_dict_path.endswith(".json"):
            act_dict = json_load(args.act_dict_path)
        else:
            act_dict = torch.load(args.act_dict_path)
        model = set_scale_and_offset(model, act_dict, "parameter" if args.lrl else None)
    #############################################################################

    print(model)

    # quantization
    if args.weight_bitwidth < 16 or args.act_bitwidth < 16:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
        if osp.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration set from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(args.calib_dataset, tokenizer=tokenizer, nsamples=args.nsamples, seed=seed, seqlen=args.seqlen)
            torch.save(dataloader, cache_dataloader)   

        if args.mode.lower() == "e2e":
            e2equant(args, model, dataloader, logger)
        else:
            omniquant(args, model, dataloader, logger, device='cuda:0')
        logger.info(time.time() - tick)

    if args.tasks:
        if args.mode.lower() == "omniquant":
            model = model.to('cuda:0')
        evaluate(args, model, tokenizer, logger)
    
    act_dict = export_act_range(model)
    json_save(osp.join(args.output_dir, 'act_dict.json'), act_dict)
    default_qcfg = export_qcfg(model)
    json_save(osp.join(args.output_dir, 'default_qcfg.json'), default_qcfg)
    model = create_fp_model(model)       
    model.save_pretrained(args.output_dir, safe_serialization=False)  
    tokenizer.save_pretrained(args.output_dir, safe_serialization=False) 


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()