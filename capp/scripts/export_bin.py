import struct
import argparse

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)


def load_hf_model(model_path):

    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    hf_config = AutoConfig.from_pretrained(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_dict = hf_model.state_dict()

    # convert LlamaConfig to ModelArgs
    config = ModelArgs()
    config.dim          = hf_config.hidden_size
    config.n_layers     = hf_config.num_hidden_layers
    config.n_heads      = hf_config.num_attention_heads
    config.n_kv_heads   = hf_config.num_key_value_heads
    config.vocab_size   = hf_config.vocab_size
    config.hidden_dim   = hf_config.intermediate_size
    config.norm_eps     = hf_config.rms_norm_eps
    config.max_seq_len  = hf_config.max_position_embeddings

    # create a new Transformer object and set weights
    model = Transformer(config)

    model.tok_embeddings.weight = nn.Parameter(hf_dict['model.embed_tokens.weight'])

    print(f"model.tok_embeddings.shape = {model.tok_embeddings.weight.shape}")

    model.eval()
    return model


def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def version1_export(model, filepath):
    """
    Export the model weights in full float32 .bin file to be read from C.
    This is same as legacy_export, but with a proper header.
    """
    version = 1

    out_file = open(filepath, 'wb')
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "ak42" in ASCII
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) write version, which will be int
    out_file.write(struct.pack('i', version))
    # 3) write the params, which will be 7 ints
    p = model.params
    print(f"model.params: {p}")

    #hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    hidden_dim = p.hidden_dim or 0
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) write some other flags
    #shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    shared_classifier = False
    out_file.write(struct.pack('B', int(shared_classifier)))
    pad = 256 - out_file.tell() # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b'\0' * pad)

    # now let's write out all the params
    weights = [
        #*[layer.attention_norm.weight for layer in model.layers],
        #*[layer.ffn_norm.weight for layer in model.layers],
        # model.norm.weight,
        model.tok_embeddings.weight,
        #*[layer.attention.wq.weight for layer in model.layers],
        #*[layer.attention.wk.weight for layer in model.layers],
        #*[layer.attention.wv.weight for layer in model.layers],
        #*[layer.attention.wo.weight for layer in model.layers],
        #*[layer.feed_forward.w1.weight for layer in model.layers],
        #*[layer.feed_forward.w2.weight for layer in model.layers],
        #*[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    # if not shared_classifier:
    #     weights.append(model.output.weight)
    for w in weights:
        serialize_fp32(out_file, w)

    # print(f"norm[:10]: {model.norm.weight.view(-1)[:10]}")
    print(f"tok_embeddings[:10]: {model.tok_embeddings.weight.view(-1)[:10]}")
    # print(f"output[:10]: {model.output.weight.view(-1)[:10]}")

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")

# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":
    # python export.py tinyllama_1.1b_chat_v0.3.bin --hf ./TinyLlama-1.1B-Chat-v0.3

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument("--hf", type=str, help="huggingface model path")
    args = parser.parse_args()

    model = load_hf_model(args.hf)

    if model is None:
        parser.error("Can't load input model!")

    # export
    version1_export(model, args.filepath)