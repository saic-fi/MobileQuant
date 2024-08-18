import os.path as osp
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from functools import partial
from datasets import load_dataset
import torch, os, argparse, gc, math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from mobilellm.model.sim_model import SimModel, SimConfig
from mobilellm.utils.bench import Evaluator, print_model_size

from mobilellm.model.hf_config import HFConfig
from mobilellm.model.hf_model import HFForCausalLM, HFRMSNorm
AutoConfig.register("hfmodel", HFConfig)
AutoModelForCausalLM.register(HFConfig, HFForCausalLM)


parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', type=str, default=None, help='path of the hf model')
parser.add_argument('--model_name', type=str, default=None, help='the size of the sim model')
parser.add_argument('--max_length', type=int, default=2048, help='max seq len')
parser.add_argument('--smooth_last', default=False, action="store_true")
parser.add_argument('--smooth_alpha', type=float, default=0.5, help='smoothquant alpha')
parser.add_argument('--calib_data', type=str, default='pileval', help='the calibration data')
parser.add_argument('--calib_path', type=str, default='data/pile/val.jsonl.zst', help='the calibration data')
parser.add_argument('--num_calib_samples', type=int, default=512, help='num of calibration samples')
parser.add_argument('--use_rand_samples', default=False, action="store_true")
# parser.add_argument("--output_dir", default='results/sim_ckpts', type=str)
args = parser.parse_args()

# args.hf_path = args.hf_path.format(args.model_name)
assert(args.hf_path is not None)
if args.hf_path.endswith('/'):
    args.hf_path = args.hf_path[:-1]
args.model_name = osp.basename(args.hf_path)
args.output_dir = args.hf_path


seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn



@torch.no_grad()
def get_last_act_scales(model, tokenizer, dataset, num_samples, max_length):
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
        if (isinstance(m, (nn.LayerNorm, HFRMSNorm)) and "model.norm" in name) or (isinstance(m, (nn.Linear, )) and "lm_head" in name):
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
        input_ids = tokenizer(line.strip(), return_tensors="pt", max_length=max_length, truncation=True).input_ids
        samples.append(input_ids)
        # following the stable diffusion demo from qualcomm, add some samples with random indices
        if args.use_rand_samples:
            rand_ids = torch.randint(tokenizer.bos_token_id+1, tokenizer.vocab_size-1, size=(1, args.max_length), dtype=torch.int32)
            samples.append(rand_ids)
    
    print("Collecting activation scales ...")
    for i in tqdm(range(len(samples))):
        model(samples[i].to(device))
        
    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def main():
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, use_fast=False, legacy=False)

    # HF model
    hf_config = AutoConfig.from_pretrained(args.hf_path)
    model_hf  = AutoModelForCausalLM.from_pretrained(args.hf_path, config=hf_config, device_map='auto', torch_dtype=torch.float32, use_safetensors=False)
    if hf_config.tie_word_embeddings:
        model_hf.lm_head.weight.data = deepcopy(model_hf.model.embed_tokens.weight.data)
    print_model_size(model_hf)

    if args.smooth_last:
        #####################################################################
        # calib data
        if args.calib_data == 'pileval':
            dataset = load_dataset("json", data_files=args.calib_path, split="train")
        elif args.calib_data == 'wikitext':
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split="train")
        else:
            raise NotImplementedError
        dataset = dataset.shuffle(seed=seed+1)
        weight_scales = model_hf.lm_head.weight.clone().abs().max(dim=0)[0]
        act_scales = get_last_act_scales(model_hf, tokenizer, dataset, args.num_calib_samples, args.max_length)["lm_head_input"].to(weight_scales.device)
        scales = (act_scales.pow(args.smooth_alpha) / weight_scales.pow(1.0-args.smooth_alpha)).clamp(min=1e-5).to(weight_scales.dtype)
        model_hf.model.norm.weight.div_(scales.view(-1))
        model_hf.lm_head.weight.mul_(scales.view(1, -1))

    # sim model
    sim_config = SimConfig.from_name(args.model_name)
    if args.max_length is not None:
        sim_config.block_size = args.max_length
    model_sim = SimModel(sim_config).to(torch.float32).cuda()
    print_model_size(model_sim)
        

    hf_state  = model_hf.state_dict()
    sim_state = model_sim.state_dict()
    new_state = {} 
    for k in list(hf_state.keys()):
        v = hf_state[k].cpu().to(torch.float32)
        k = k.replace('model.', '')
        if sim_config.impl_sym_pch_as_slinear and 'w2.weight' in k:
            orig_weight = v
            scale = torch.max(torch.abs(orig_weight), dim=1)[0]
            new_weight = orig_weight / scale.unsqueeze(1)
            scale_k = k.replace('w2.weight', 'w2.scale')
            new_state[scale_k] = scale
            weight_k = k.replace('w2.weight', 'w2.linear.weight')
            new_state[weight_k] = new_weight
        if "lm_head" in k:
            orig_weight = v
            scale = torch.max(torch.abs(orig_weight), dim=1)[0]
            new_weight = orig_weight / scale.unsqueeze(1)
            scale_k = k.replace('lm_head.weight', 'lm_head.scale')
            new_state[scale_k] = scale
            weight_k = k.replace('lm_head.weight', 'lm_head.linear.weight')
            new_state[weight_k] = new_weight
        if "norm.weight" in k:
            new_state[k] = v * math.sqrt(sim_config.n_embd)
            # print(k, torch.amin(v), torch.amax(v), torch.amin(new_state[k]), torch.amax(new_state[k]))
        elif "gemma" in args.model_name.lower() and "embed_tokens" in k:
            new_state[k] = v * math.sqrt(sim_config.n_embd)
        elif k in sim_state:
            new_state[k] = v
    msg = model_sim.load_state_dict(new_state, strict=True)
    print(msg)

    # fuse the scaling factor into q_proj
    for layer in model_sim.layers:
        layer.self_attn.q_proj.weight.data.div_(math.sqrt(sim_config.head_dim))

    out_states = {}
    for k, v in model_sim.state_dict().items():
        out_states[k] = v.cpu()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(out_states, os.path.join(args.output_dir, 'sim_{}.pth'.format(args.model_name)))
    
    
if __name__ == '__main__':
    main()
