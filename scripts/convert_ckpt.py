import contextlib, gc, json, sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch

import os.path as osp
from safetensors import SafetensorError
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from mobilellm.model.hf_config import HFConfig 
from mobilellm.model.hf_model import HFForCausalLM
from mobilellm.utils.io import json_load, json_save
from transformers import AutoModelForCausalLM, AutoConfig


WEIGHT_RENAME_MAPS = {
    "llama": { "gate_proj": "w1", "down_proj": "w2", "up_proj": "w3" },
    "mistral": { "gate_proj": "w1", "down_proj": "w2", "up_proj": "w3" },
    "gemma": { "gate_proj": "w1", "down_proj": "w2", "up_proj": "w3" },
    "phi": { "fc1": "w1", "fc2": "w2", "dense": "o_proj", "final_layernorm": "norm"},
    "stablelm": { "gate_proj": "w1", "down_proj": "w2", "up_proj": "w3" },
    "qwen2": { "gate_proj": "w1", "down_proj": "w2", "up_proj": "w3" },
    "mixtral": {"block_sparse_moe": "mlp"},
}


def rename_weight(hf_weights, rename_map):
    for src_name in list(hf_weights.keys()):
        tgt_name = src_name
        for a, b in rename_map.items():
            if a in src_name:
                tgt_name = src_name.replace(a, b)
                break 
        if tgt_name != src_name:
            hf_weights[tgt_name] = hf_weights[src_name]
            hf_weights.pop(src_name)
    return hf_weights


def preprocess_weight(hf_weights, model_type):
    if model_type != "gemma":
        return hf_weights
    for name in list(hf_weights.keys()):
        if "norm" in name:
            hf_weights[name] = hf_weights[name] + 1
    return hf_weights



def rename_index(index_file, rename_map):
    for src_name in list(index_file["weight_map"].keys()):
        tgt_name = src_name
        for a, b in rename_map.items():
            if a in src_name:
                tgt_name = src_name.replace(a, b) 
                break
        if tgt_name != src_name:
            index_file["weight_map"][tgt_name] = index_file["weight_map"][src_name]
            index_file["weight_map"].pop(src_name)
    return index_file


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/llama/Llama-2-7b-hf"),
    output_dir: Path,
) -> None:

    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    rename_map = WEIGHT_RENAME_MAPS[config.model_type]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = checkpoint_dir / "pytorch_model.bin.index.json"
    safetensors_map_json_path = checkpoint_dir / "model.safetensors.index.json"

    if safetensors_map_json_path.is_file():
        index_file = json_load(safetensors_map_json_path)
        ckpt_files = set([checkpoint_dir / sth for sth in index_file["weight_map"].values()])
        index_file = rename_index(index_file, rename_map)
        json_save(output_dir / "model.safetensors.index.json", index_file)
    elif pytorch_bin_map_json_path.is_file():  # not all checkpoints have this file
        index_file = json_load(pytorch_bin_map_json_path)
        ckpt_files = set([checkpoint_dir / bin for bin in index_file["weight_map"].values()])
        index_file = rename_index(index_file, rename_map)
        json_save(output_dir / "pytorch_model.bin.index.json", index_file)
    else:
        ckpt_files = set(checkpoint_dir.glob("*.bin"))
        ckpt_files.update(set(checkpoint_dir.glob("*.safetensors")))
    if not ckpt_files:
        raise ValueError(f"Expected {str(checkpoint_dir)!r} to contain .bin/.safetensors files")

    for ckpt_file in sorted(ckpt_files):
        print("Processing", ckpt_file)
        if osp.splitext(ckpt_file)[1] == ".bin":
            hf_weights = torch.load(ckpt_file)
            hf_weights = rename_weight(hf_weights, rename_map)
            hf_weights = preprocess_weight(hf_weights, config.model_type)
            torch.save(hf_weights, output_dir / osp.basename(ckpt_file))
        else:
            hf_weights = safetensors_load(ckpt_file)
            hf_weights = rename_weight(hf_weights, rename_map)
            hf_weights = preprocess_weight(hf_weights, config.model_type)
            safetensors_save(hf_weights, output_dir / osp.basename(ckpt_file), metadata={"format": "pt"})


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_hf_checkpoint, as_positional=False)
