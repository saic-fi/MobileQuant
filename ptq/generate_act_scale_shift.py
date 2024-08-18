import torch
import torch.nn as nn
import os.path as osp
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
import os, argparse, gc, functools

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from mobilellm.utils.bench import print_model_size
from mobilellm.utils.io import json_load, json_save
from mobilellm.model.hf_model import HFForCausalLM, HFRMSNorm, HFSkipRMSNorm, HFDecoderLayer
from mobilellm.model.hf_config import HFConfig


AutoConfig.register("hfmodel", HFConfig)
AutoModelForCausalLM.register(HFConfig, HFForCausalLM)


parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', type=str, help='path of the hf model')
parser.add_argument('--seq_len', type=int, default=4096, help='max seq len')
parser.add_argument('--calib_data', type=str, default='pileval', help='the calibration data')
parser.add_argument('--calib_path', type=str, default='data/pile/val.jsonl.zst', help='the calibration data')
parser.add_argument('--num_samples', type=int, default=512, help='num of calibration samples')
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
def get_act_scales(model, tokenizer, dataset, num_samples, seq_len):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, field, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        name_field = "{}_{}".format(name, field)
        if name_field in act_scales:
            act_scales[name_field] = torch.max(act_scales[name_field], comming_max)
        else:
            act_scales[name_field] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):    
            x = x[0]
        if isinstance(y, tuple):    
            y = y[0]
        stat_tensor(name, 'input', x)
        stat_tensor(name, 'output', y)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.LayerNorm, HFRMSNorm)):
            hooks.append(
                m.register_forward_hook(
                    partial(stat_input_hook, name=name))
            )

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
    
    print("Collecting activation scales ...")
    for i in tqdm(range(len(samples))):
        model(samples[i].to(device))
        
    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def get_act_shifts(model, tokenizer, dataset, num_samples, seq_len):
    model.eval()
    device = next(model.parameters()).device
    act_shifts = {}

    def stat_tensor(name, field, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        comming_min = torch.min(tensor, dim=0)[0].float().cpu()
        name_field = "{}_{}".format(name, field)
        if name_field in act_shifts:
            act_shifts[name_field] = 0.99*act_shifts[name_field] + 0.01 *((comming_max+comming_min)/2)
        else:
            act_shifts[name_field] = (comming_max+comming_min)/2

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):    
            x = x[0]
        if isinstance(y, tuple):    
            y = y[0]
        stat_tensor(name, 'input', x)
        stat_tensor(name, 'output', y)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.LayerNorm, HFRMSNorm)):
            hooks.append(
                m.register_forward_hook(
                    partial(stat_input_hook, name=name))
            )

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
    
    print("Collecting activation shifts ...")
    for i in tqdm(range(len(samples))):
        model(samples[i].to(device))
        
    for h in hooks:
        h.remove()

    return act_shifts

    
@torch.no_grad()
def main():
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, use_fast=False, legacy=False)
    config = AutoConfig.from_pretrained(args.hf_path, trust_remote_code=True)
    config.use_matmul_as_module = True
    config._attn_implementation = "eager"
    config.l2norm_as_rmsnorm = True
    model = AutoModelForCausalLM.from_pretrained(args.hf_path, config=config, device_map='auto', torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation="eager")
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

    act_scales = get_act_scales(model, tokenizer, dataset, args.num_samples, args.seq_len)
    torch.save(act_scales, osp.join(args.output_dir, 'act_scales.pth'))

    act_shifts = get_act_shifts(model, tokenizer, dataset, args.num_samples, args.seq_len)
    torch.save(act_shifts, osp.join(args.output_dir, 'act_shifts.pth'))


if __name__ == '__main__':
    main()
