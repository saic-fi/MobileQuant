import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from glob import glob
import onnx, onnxruntime
from copy import deepcopy
from functools import partial
from datasets import load_dataset
import shutil, subprocess, re, zlib, subprocess
import torch, os, argparse, json, logging, sys, gc, math
from lm_eval import evaluator, tasks, utils
from lm_eval.base import BaseLM


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedModel


from mobilellm.model.sim_model import SimConfig, SimModel, create_conv_model, Sim_Head, Sim_QNN
from mobilellm.utils.bench import print_model_size
from mobilellm.utils.io import json_load, json_save
from device.utils import incorporate_l2norm, update_qcfg_sim, to_device


from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_common.utils import AimetLogger
AimetLogger.set_level_for_all_areas(logging.INFO)


from mobilellm.model.hf_config import HFConfig
from mobilellm.model.hf_model import HFForCausalLM
AutoConfig.register("hfmodel", HFConfig)
AutoModelForCausalLM.register(HFConfig, HFForCausalLM)


parser = argparse.ArgumentParser()

parser.add_argument('--hf_path', type=str, default=None, help='path of the hf model')
parser.add_argument('--model_name', type=str, default=None)
# parser.add_argument('--model_path', type=str, default='results/sim_ckpts/sim_{}.pth')
parser.add_argument('--max_length', type=int, default=2048, help='max seq len for the samples')
parser.add_argument('--calib_data', type=str, default='pileval', help='the calibration data')
parser.add_argument('--calib_path', type=str, default='data/pile/val.jsonl.zst', help='the calibration data')
parser.add_argument('--num_calib_samples', type=int, default=512, help='num of calibration samples')
parser.add_argument('--use_rand_samples', default=False, action="store_true")
parser.add_argument('--ctx_encoding', type=str, default=None, help='the encoding config file')
parser.add_argument("--output_dir", default='results/sim_{}_eval_ctx', type=str)
parser.add_argument('--default_config', type=str, default='assets/aimet_config.json', help='the default config file')
parser.add_argument('--num_blocks', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", default="wikitext", type=str, help='llm task from the harness repo')
parser.add_argument('--weight_bitwidth', type=int, default=8)
parser.add_argument('--act_bitwidth', type=int, default=8)
parser.add_argument('--outlier_act_bitwidth', type=int, default=16)
parser.add_argument('--use_conv', default=False, action="store_true")
parser.add_argument('--per_channel', default=False, action="store_true")


args = parser.parse_args()
assert(args.hf_path is not None)
if args.hf_path.endswith('/'):
    args.hf_path = args.hf_path[:-1]
args.model_name = osp.basename(args.hf_path)
args.output_dir = args.output_dir.format(args.model_name)
args.model_path = osp.join(args.hf_path, f"sim_{args.model_name}.pth")


if args.per_channel:
    args.default_config = "assets/aimet_per_channel_config.json"


seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

from aimet_torch.qc_quantize_op import QcQuantizeWrapper
def disable_quant_sim(sim_model):
    for name, module in sim_model.model.named_modules():
        if isinstance(module, QcQuantizeWrapper):
            # if name.startswith('module_matmul'):
            #     if name == "module_matmul":
            #         for i in range(len(module.input_quantizers)):
            #             module.input_quantizers[i].enabled = False
            #         for i in range(len(module.output_quantizers)):
            #             module.output_quantizers[i].enabled = False
            #     else:
            #         ind = name.split('_')[-1]
            #         assert ind.isdigit()
            #         ind = int(ind)
            #         if ind % 2 == 0:
            #             for i in range(len(module.input_quantizers)):
            #                 module.input_quantizers[i].enabled = False
            #             for i in range(len(module.output_quantizers)):
            #                 module.output_quantizers[i].enabled = False
            if not (name.startswith('module_matmul') or name.startswith("module_add") or "module_reshape" in name or "module_transpose" in name or "softmax" in name or "sigmoid" in name or "mlp.act" in name or "module_cat" in name or "module_mul" in name or "norm.module_mul" in name or "module_normalize" in name or "q_proj" in name or "k_proj" in name or "v_proj" in name or "w1" in name or "w3" in name or "o_proj" in name or "w2" in name):
                print("not quantizing", name)
                for i in range(len(module.input_quantizers)):
                    module.input_quantizers[i].enabled = False
                for i in range(len(module.output_quantizers)):
                    module.output_quantizers[i].enabled = False
                for i in list(module.param_quantizers.keys()):
                    module.param_quantizers[i].enabled = False
            
    return sim_model


class Evaluator:
    def __init__(self, dataset, tokenizer, max_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    @torch.no_grad()
    def evaluate(self, model, model_head):
        model.eval()
        device = next(model.parameters()).device
        dtype  = next(model.parameters()).dtype
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latency = 0

        if not isinstance(model, PreTrainedModel):
            attention_mask = SimModel._make_causal_mask(self.max_length, self.max_length, self.max_length)
            attention_mask = model_head.config.neg_inf * attention_mask
            attention_mask = attention_mask.to(dtype).to(device)
            position_ids = torch.arange(0, self.max_length, dtype=torch.long).to(device)

        for batch in tqdm(self.dataset):
            input_ids = torch.tensor(self.tokenizer(batch['text']).input_ids)[:self.max_length]
            input_ids = input_ids.to(device).unsqueeze(0)
            label = input_ids[:, -1]
            pad_len = self.max_length - input_ids.shape[1]
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)

            torch.cuda.synchronize()
            start.record()
            if isinstance(model, PreTrainedModel):
                outputs = model(input_ids, attention_mask=None)
            else:
                ctx_input = model_head(input_ids[0], attention_mask=attention_mask, position_ids=position_ids)
                outputs = model(*ctx_input)[0]
            end.record()
            torch.cuda.synchronize()
            latency += start.elapsed_time(end)
            if isinstance(model, PreTrainedModel):
                logits = outputs.logits
            else:
                logits = outputs
                logits = logits.unsqueeze(0)
            last_token_logits = logits[:, -2-pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()

        acc = hit / total
        latency = latency / len(self.dataset)
        return acc, latency


class LMEvalAdaptor(BaseLM):
    def __init__(self, 
            model_name, 
            model_head, 
            model_ctx, 
            config, 
            tokenizer, 
            batch_size=1, 
            max_length=-1
        ):
        super().__init__()
        assert isinstance(batch_size, int)
        self.model_name = model_name
        self.model_head = model_head
        self.model = model_ctx 
        self.config = config 
        self.model.eval()
        self.model_head.eval()
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = batch_size
        self._max_length = max_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length != -1:
            return self._max_length
        if hasattr(self.model.config, 'n_ctx'):
            return self.model.config.n_ctx
        elif hasattr(self.model.config, 'max_position_embeddings'):
            return self.model.config.max_position_embeddings
        elif hasattr(self.model.config, 'n_positions'):
            return self.model.config.n_positions
        elif 'bloom' in self.model_name:
            return 2048
        elif 'llama' in self.model_name:
            return 2048  # TODO: did not check this
        elif 'mpt' in self.model_name:
            return 2048
        elif 'falcon' in self.model_name:
            return 2048
        else:
            print(self.model.config)
            raise NotImplementedError

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        with torch.no_grad():
            inps = inps[:,:self.max_length]
            cur_len = inps.shape[1]
            if cur_len < self.max_length:
                pad_len = self.max_length - inps.shape[1]
                inps = torch.nn.functional.pad(inps, (0, pad_len), value=0)
            
            attention_mask = SimModel._make_causal_mask(self.max_length, self.max_length, self.max_length)
            attention_mask = self.config.neg_inf * attention_mask
            attention_mask = attention_mask.to(self.device)
            position_ids = torch.arange(0, self.max_length, dtype=torch.long).to(self.device)
            kwargs = {'attention_mask': attention_mask, 'position_ids': position_ids}

            # max_position_embeddings = self.config.block_size
            # causal_mask = SimModel._make_causal_mask(max_position_embeddings, max_position_embeddings, max_position_embeddings)
            # attention_mask = self.config.neg_inf * causal_mask
            # position_ids = torch.arange(0, max_position_embeddings, dtype=torch.long)
            # context_ids = inps[0]
            # context_len = len(context_ids)
            # device = next(self.model_head.parameters()).device
            # input_ids = torch.cat(
            #     [
            #         context_ids,
            #         torch.zeros((max_position_embeddings - context_len), dtype=torch.long, device=context_ids.device),
            #     ]
            # )

            input_feats, attention_mask, cos, sin = self.model_head(inps[0], **kwargs)
            logits, k_cache, v_cache = self.model(input_feats, attention_mask, cos, sin)
            out = logits[:cur_len].unsqueeze(0)
            return out

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context,
            max_new_tokens=self.max_gen_toks,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False
        )


def main():
    #####################################################################
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, use_fast=False, legacy=False, trust_remote_code=True)
    #####################################################################

    #####################################################################
    # sim model
    config = SimConfig.from_name(args.model_name)
    config.block_size = args.max_length
    model_ori = SimModel(config)
    for x in model_ori.parameters(): 
        x.requires_grad = False
    ckpt = torch.load(args.model_path, map_location='cpu')
    msg = model_ori.load_state_dict(ckpt, strict=True)
    print(msg)
    if args.use_conv:
        model_ori = create_conv_model(model_ori)

    #####################################################################
    # sub modules
    if args.num_blocks is None:
        args.num_blocks = config.n_layer
    model_head = Sim_Head.from_sim(model_ori).cuda()
    model_body = Sim_QNN.from_sim(model_ori, args.num_blocks).cuda()
    device = next(model_body.parameters()).device
    model_head.eval()
    model_body.eval()
    del model_ori
    gc.collect()
    torch.cuda.empty_cache()

    #####################################################################
    # calib data
    if args.calib_data == 'pileval':
        dataset = load_dataset("json", data_files=args.calib_path, split="train")
    elif args.calib_data == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split="train")
    else:
        dataset = load_dataset(args.calib_data, split="validation")
    dataset = dataset.shuffle(seed=seed+1)
    samples = []
    position_ids = torch.arange(0, args.max_length, dtype=torch.long)
    for i in tqdm(range(args.num_calib_samples)):
        if "text" in dataset[i]:        line = dataset[i]["text"]
        elif "content" in dataset[i]:   line = dataset[i]["content"]
        elif "ctx" in dataset[i]:       line = dataset[i]["ctx"]
        else:                           raise NotImplementedError
        input_ids = tokenizer(line.strip(), return_tensors="pt", max_length=args.max_length, truncation=True).input_ids[0]
        valid_len = input_ids.shape[0]
        attention_mask = SimModel._make_causal_mask(args.max_length, args.max_length, args.max_length)
        attention_mask = config.neg_inf * attention_mask
        # pad to seq_len
        input_ids = torch.nn.functional.pad(input_ids, (0, args.max_length - input_ids.shape[0]), value=tokenizer.eos_token_id)
        samples.append((input_ids, attention_mask, position_ids))
        if args.use_rand_samples:
            rand_ids       = torch.randint(tokenizer.bos_token_id+1, tokenizer.vocab_size-1, size=(args.max_length,), dtype=torch.int32)
            attention_mask = SimModel._make_causal_mask(args.max_length, args.max_length, args.max_length)
            attention_mask = config.neg_inf * attention_mask
            samples.append((rand_ids, attention_mask, position_ids))

    #####################################################################
    # prepare model
    model_ctx = prepare_model(model_body, concrete_args={'k_cache': None, 'v_cache': None})

    os.makedirs(args.output_dir, exist_ok=True)
    ctx_dir = osp.join(args.output_dir, 'ctx')
    os.makedirs(ctx_dir, exist_ok=True)


    #####################################################################
    # sample input and output
    with torch.no_grad():
        ctx_sample = model_head(*to_device(samples[0], device))
        fp_outputs = model_ctx(*ctx_sample)[0]
    #####################################################################


    #####################################################################
    ## Sim model
    sim_ctx = QuantizationSimModel(
        model=model_ctx, 
        quant_scheme=QuantScheme.post_training_tf, 
        dummy_input=ctx_sample,
        rounding_mode='nearest',
        in_place=True,
        config_file=args.default_config,
        default_output_bw=args.act_bitwidth,
        default_param_bw=args.weight_bitwidth,
        default_data_type=QuantizationDataType.int,
    )

    #####################################################################
    ## 16-bit
    # sixteen_bit_output_activations = ['module_normalize', 'o_proj', 'w2', 'lm_head', 'softmax', 'sigmoid', 'mlp.act']
    # sixteen_bit_input_activations = ['module_normalize', 'norm.module_mul', 'w2', 'lm_head', 'softmax', 'sigmoid', 'mlp.act']
    sixteen_bit_output_activations = ['module_normalize', 'o_proj', 'w2', 'lm_head', 'softmax']
    sixteen_bit_input_activations = ['module_normalize', 'norm.module_mul', 'w2', 'lm_head', 'softmax']
    sim_ctx = update_qcfg_sim(sim_ctx, sixteen_bit_input_activations, sixteen_bit_output_activations, config, args.num_blocks, args.outlier_act_bitwidth)
    #####################################################################
    ## Calibration
    #####################################################################
    @torch.no_grad()
    def pass_ctx_calibration_data(sim_model, *inargs):
        sim_model.eval()
        with torch.no_grad():
            for i in tqdm(range(len(samples))):
                input4 = model_head(*to_device(samples[i], device))
                sim_model(*input4)
    #####################################################################
    sim_ctx.model.config = config
    if args.ctx_encoding is not None:
        load_encodings_to_sim(sim_ctx, args.ctx_encoding)
    else:
        sim_ctx.compute_encodings(forward_pass_callback=pass_ctx_calibration_data, forward_pass_callback_args=None)
    del samples, dataset
    gc.collect()
    torch.cuda.empty_cache()

    # sim_ctx = disable_quant_sim(sim_ctx)

    #####################################################################
    ## Harness
    task_names = args.tasks.split(",")
    task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
    print(f"Running tasks: {task_names}")
    lm_eval_model = LMEvalAdaptor(args.model_name, 
        model_head, sim_ctx.model, config,
        tokenizer, args.batch_size, max_length=config.block_size
    )
    # lm_eval_model = LMEvalAdaptor(args.model_name, 
    #     model_head, model_body, config,
    #     tokenizer, args.batch_size, max_length=config.block_size
    # )
    results = evaluator.simple_evaluate(model=lm_eval_model, tasks=task_names, batch_size=args.batch_size, no_cache=True, num_fewshot=0)
    print(evaluator.make_table(results))
    results["config"]["model"] = args.model_name
    os.makedirs(args.output_dir, exist_ok=True)
    json_save(osp.join(args.output_dir, 'results.json'), results)
    

if __name__ == '__main__':
    main()
