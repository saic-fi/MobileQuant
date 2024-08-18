import torch
import torch.nn as nn
import numpy as np
import os.path as osp
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
import os, argparse, gc, functools
from collections import defaultdict
from transformers.activations import GELUActivation, NewGELUActivation, PytorchGELUTanh


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from mobilellm.utils.bench import print_model_size
from mobilellm.utils.io import json_load, json_save
from mobilellm.model.hf_model import HFForCausalLM, HFRMSNorm, HFSkipRMSNorm, HFDecoderLayer
from mobilellm.model.hf_config import HFConfig
from mobilellm.quantization.qmodule import QRMSNorm, QLinear, QMatMul
from mobilellm.model.ops import L2Norm, ElementwiseMul, ElementwiseAdd, FMatMul

AutoConfig.register("hfmodel", HFConfig)
AutoModelForCausalLM.register(HFConfig, HFForCausalLM)


parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', type=str, help='path of the hf model')
parser.add_argument('--seq_len', type=int, default=4096, help='max seq len')
parser.add_argument('--calib_data', type=str, default='pileval', help='the calibration data')
parser.add_argument('--calib_path', type=str, default='data/pile/val.jsonl.zst', help='the calibration data')
parser.add_argument('--num_samples', type=int, default=512, help='num of calibration samples')
parser.add_argument('--per_channel', default=False, action="store_true")
parser.add_argument('--use_rand_samples', default=False, action="store_true")
parser.add_argument("--output_dir", default=None, type=str)
args = parser.parse_args()


if args.output_dir is None:
    args.output_dir = args.hf_path


seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


@torch.no_grad()
def get_act_range(model, tokenizer, dataset, num_samples, seq_len):
    model.eval()
    device = next(model.parameters()).device
    act_dict = defaultdict(dict)


    def update_act_range(name, field, inp, per_channel):
        if args.per_channel:
            inp = inp.reshape((-1, inp.shape[-1]))
            min_x, max_x = torch.min(inp, dim=0)[0], torch.max(inp, dim=0)[0]
            if name not in act_dict or field not in act_dict[name]:
                act_dict[name][field] = torch.stack((min_x, max_x), dim=0)
            else:
                act_dict[name][field][0] = torch.minimum(act_dict[name][field][0], min_x)
                act_dict[name][field][1] = torch.maximum(act_dict[name][field][1], max_x)
        else:
            min_x, max_x = inp.min().item(), inp.max().item()
            if name not in act_dict or field not in act_dict[name]:
                act_dict[name][field] = [min_x, max_x]
            else:
                act_dict[name][field] = [min(act_dict[name][field][0], min_x), max(act_dict[name][field][1], max_x)]


    def stat_io_hook(m, xx, yy, name):
        # input
        x = xx[0] if isinstance(xx, tuple) else xx
        x = x.detach()

        update_act_range(name, "input", x, args.per_channel)
        
        # output
        y = yy[0] if isinstance(yy, tuple) else yy
        y = y.detach()

        update_act_range(name, "output", y, args.per_channel)

        # the second input for matmul
        if isinstance(m, (FMatMul, )):
            update_act_range(name, "input2", xx[1].detach(), args.per_channel)

        # print('shape', name, x.shape, y.shape, act_dict[name]["input"].shape, act_dict[name]["output"].shape)
    

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.SiLU, nn.Softmax, GELUActivation, NewGELUActivation, PytorchGELUTanh, nn.LayerNorm, HFRMSNorm, FMatMul)) or 'attn_quantizer' in name or 'softmax_quantizer' in name:
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    samples = []
    for i in tqdm(range(num_samples)):
        if "text" in dataset[i]:        line = dataset[i]["text"]
        elif "content" in dataset[i]:   line = dataset[i]["content"]
        elif "ctx" in dataset[i]:       line = dataset[i]["ctx"]
        else:                           raise NotImplementedError
        input_ids = tokenizer(line.strip(), return_tensors="pt", max_length=seq_len, truncation=True).input_ids
        samples.append(input_ids)
        # following the stable diffusion demo from qualcomm, add some samples with random indices
        if args.use_rand_samples:
            rand_ids = torch.randint(tokenizer.bos_token_id+1, tokenizer.vocab_size-1, size=(1, args.seq_len), dtype=torch.int32)
            samples.append(rand_ids)

    print("Collecting activation scales for quantization...")
    pbar = tqdm(range(len(samples)))
    for i in pbar:
        model(samples[i].to(device))
        if args.per_channel:
            mean_scale = np.mean([torch.mean(v["input"].to(torch.float32)).item() for v in act_dict.values()])
        else:
            mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()

    return act_dict


@torch.no_grad()
def main():
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, use_fast=False, legacy=False)
    config = AutoConfig.from_pretrained(args.hf_path, trust_remote_code=True)
    config.use_matmul_as_module = True
    config._attn_implementation = "eager"
    config.l2norm_as_rmsnorm = True
    model = AutoModelForCausalLM.from_pretrained(args.hf_path, config=config, device_map='auto', torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="eager", 
        use_safetensors=False, # this is important, it gurantees that we use the smoothed model (saved as bin files) to compute the range
    )
    print(model)
    print_model_size(model, False)  

    # calib data
    if args.calib_data == 'pileval':
        dataset = load_dataset("json", data_files=args.calib_path, split="train")
    elif args.calib_data == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split="train")
    elif args.calib_data == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    else:
        dataset = load_dataset(args.calib_data, split="validation")

    dataset = dataset.shuffle(seed=seed)
    os.makedirs(args.output_dir, exist_ok=True)

    act_dict = get_act_range(model, tokenizer, dataset, args.num_samples, args.seq_len)
    if not args.per_channel:
        json_save(osp.join(args.output_dir, 'act_dict.json'), act_dict)
    else:
        for k in list(act_dict.keys()):
            act_dict[k]["input"] = act_dict[k]["input"].cpu()
            act_dict[k]["output"] = act_dict[k]["output"].cpu()
        torch.save(act_dict, osp.join(args.output_dir, 'act_dict_per_channel.pth'))
    

if __name__ == '__main__':
    main()
