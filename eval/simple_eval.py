import os.path as osp
from datasets import load_dataset
import torch, os, argparse, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from mobilellm.model.hf_config import HFConfig 
from mobilellm.model.hf_model  import HFForCausalLM
from mobilellm.model.sim_model import SimConfig, SimModel
from mobilellm.utils.io import json_load, json_save
from mobilellm.utils.bench import print_model_size, Evaluator

AutoConfig.register("hfmodel", HFConfig)
AutoModelForCausalLM.register(HFConfig, HFForCausalLM)


parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', type=str, default=None, help='path of the hf model')
parser.add_argument('--max_length', type=int, default=2048, help='max seq len')
parser.add_argument('--mode', type=str, default="hf", choices=["hf", "w4a16", "custom", "int", "sim"])
parser.add_argument('--act_dict_path', type=str, default=None, help='the act dict file for custom quantization')
parser.add_argument('--override_qcfg_path', type=str, default=None, help='the fine-grained config for custom quantization')
parser.add_argument('--group_size', type=int, default=-1, help='group size if we like to use per-group quantization')
# parser.add_argument('--use_matmul_as_module', default=False, action="store_true", help='whether to use qmatmul')
args = parser.parse_args()


if args.mode == "custom":
    if args.act_dict_path is None:
        args.act_dict_path = osp.join(args.hf_path, "act_dict.json")
    if args.override_qcfg_path is None:
        args.override_qcfg_path = osp.join(args.hf_path, "default_qcfg.json")


seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


@torch.inference_mode()
def main():
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, use_fast=False, legacy=False, trust_remote_code=True)
    dataset   = load_dataset('lambada', split='validation[:1000]')
    evaluator = Evaluator(dataset, tokenizer, args.max_length)
    config = AutoConfig.from_pretrained(args.hf_path, trust_remote_code=True)
    # config.use_matmul_as_module = args.use_matmul_as_module
    config.use_matmul_as_module = True
    config._attn_implementation = "eager"
    config.l2norm_as_rmsnorm = True
    model_name = osp.basename(args.hf_path)

    if args.mode == "sim":
        model_path = osp.join(args.hf_path, f'sim_{model_name}.pth')
        sim_config = SimConfig.from_name(model_name)
        sim_config.block_size = args.max_length
        model = SimModel(sim_config).cuda()
        ckpt = torch.load(model_path, map_location='cpu')
        msg = model.load_state_dict(ckpt, strict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_path, 
            config=config, 
            device_map='auto', 
            torch_dtype=torch.float32, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True, 
            attn_implementation="eager", 
            use_safetensors=args.mode != "custom"
        ) 

    if args.mode in ["w4a16", "custom"]:
        from mobilellm.quantization.qmodule import QuantConfig, create_sim_qmodel, create_weight_only_qmodel, set_scale_and_offset, update_qcfg
        if args.mode == "w4a16":
            w4_qcfg  = QuantConfig(bitwidth=4, group_size=args.group_size, is_per_channel=True)
            model = create_weight_only_qmodel(model, w4_qcfg)
            print(f"Evaluating w4a16 model...")
        elif args.mode == "custom" and args.act_dict_path is not None and args.override_qcfg_path is not None:
            model = create_sim_qmodel(model)
            override_qcfg = json_load(args.override_qcfg_path)
            model = update_qcfg(model, override_qcfg)
            # pre-computed activation range
            if args.act_dict_path.endswith(".json"):
                act_dict = json_load(args.act_dict_path)
            else:
                act_dict = torch.load(args.act_dict_path)
            model = set_scale_and_offset(model, act_dict, 'parameter')
            print(f'Evaluating the custom quantized model...')
        else:
            raise NotImplementedError

    model.eval()
    print(model)
    print_model_size(model, False)

    
    with torch.no_grad():
        acc_fp, latency_fp = evaluator.evaluate(model)
    print(f'Accuracy: {acc_fp:.3f}, per-sample latency: {latency_fp:.3f}ms')


if __name__ == '__main__':
    main()
