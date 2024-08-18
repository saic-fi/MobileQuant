import torch, gc
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from dataclasses import dataclass
from typing_extensions import Self, Optional
from .ops import ElementwiseMul, ElementwiseAdd, FMatMul
from .sim_layers import FROPE, FRMSNorm, SLinear


@dataclass
class SimConfig:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: int = 32
    n_embd: int = 4096
    head_dim: int = 128
    intermediate_size: int = 11008

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    norm_eps: float = 1e-05
    for_aimet: bool = False
    neg_inf: int = -40000
    use_sha: bool = False

    attention_bias: bool = False
    mlp_bias: bool = False
    norm_layer: str = 'rmsnorm'
    act_fn: str = "silu"
    impl_sym_pch_as_slinear: bool = False


    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**th_configs[name])


th_configs = {    
    "llama-1.1b-w4a8": dict(n_layer=22, n_head=32, n_kv_head=4, head_dim=64, n_embd=2048, intermediate_size=5632, vocab_size=32000, block_size=1024, norm_eps=1e-5),
    "llama-1.1b-w8a8": dict(n_layer=22, n_head=32, n_kv_head=4, head_dim=64, n_embd=2048, intermediate_size=5632, vocab_size=32000, block_size=1024, norm_eps=1e-5, impl_sym_pch_as_slinear=True),
    "gemma-2b-w4a8": dict(n_layer=18, n_head=8, n_kv_head=1, head_dim=256, n_embd=2048, intermediate_size=16384, vocab_size=256000, block_size=1024, norm_eps=1e-6, act_fn="gelu"),
    "gemma-2b-w8a8": dict(n_layer=18, n_head=8, n_kv_head=1, head_dim=256, n_embd=2048, intermediate_size=16384, vocab_size=256000, block_size=1024, norm_eps=1e-6, act_fn="gelu", impl_sym_pch_as_slinear=True)
}


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv(x.unsqueeze(-1).unsqueeze(-1))
        x = x.squeeze(-1).squeeze(-1)
        return x

    @staticmethod
    def from_linear(module):
        out = Conv(module.in_features, module.out_features, bias=module.bias is not None)
        with torch.no_grad():
            out.conv.weight.copy_(module.weight.unsqueeze(-1).unsqueeze(-1))
            if out.conv.bias is not None:
                out.conv.bias.copy_(module.bias)
        return out

    @staticmethod
    def to_linear(module):
        out = nn.Linear(module.conv.in_channels, module.conv.out_channels, bias=module.conv.bias is not None)
        with torch.no_grad():
            out.weight.copy_(module.conv.weight.squeeze(-1).squeeze(-1))
            if out.bias is not None:
                out.bias.copy_(module.conv.bias)
        return out


def create_conv_model(model):
    for name, module in reversed(model._modules.items()):
        if isinstance(module, nn.Linear):
            model._modules[name] = Conv.from_linear(module)
        elif len(list(module.children())) > 1:
            create_conv_model(module)
    return model


class SimModel(nn.Module):
    def __init__(self, config: SimConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList(SimBlock(config) for _ in range(config.n_layer))
        if config.norm_layer.lower() == "rmsnorm":
            self.norm = FRMSNorm(config)
        else:
            raise NotImplementedError
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.mlp_bias)
        self.lm_head = SLinear(config.n_embd, config.vocab_size, bias=config.mlp_bias, for_aimet=config.for_aimet)


        # rope
        cos, sin = SimModel._build_rope_cache(config.block_size, n_elem=config.n_embd // config.n_head)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, 
            input_ids:      torch.LongTensor, 
            attention_mask: torch.Tensor, 
            position_ids:   torch.LongTensor,
            k_cache:        Optional[torch.Tensor] = None,
            v_cache:        Optional[torch.Tensor] = None,
        ):
        # input_ids: (q_len)
        # attention_mask: (1, q_len, kv_len)
        # position_ids: (q_len)
        # k_cache: (num_blocks, n_heads, T-1, head_dim)
        # v_cache: (num_blocks, n_heads, T-1, head_dim)
        x = self.embed_tokens(input_ids).view(-1, self.config.n_embd)  # token embeddings of shape (t, n_embd)
        cos, sin = self.cos[position_ids].view(-1, self.config.head_dim), self.sin[position_ids].view(-1, self.config.head_dim)
        out_k, out_v = [], []
        for i, block in enumerate(self.layers):
            if k_cache is not None:
                x, k, v = block(x, attention_mask, cos, sin, k_cache[i], v_cache[i]) 
            else:
                x, k, v = block(x, attention_mask, cos, sin) # (t, n_embd)
            out_k.append(k)
            out_v.append(v)
        out_k = torch.stack(out_k, 0)
        out_v = torch.stack(out_v, 0)
        x = self.norm(x) # (t, n_embd)
        logits = self.lm_head(x)  # (t, vocab_size)
        return logits, out_k, out_v

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(SimConfig.from_name(name))

    @staticmethod
    def _make_causal_mask(q_len, kv_seq_len, valid_len):
        assert (kv_seq_len >= valid_len) and (kv_seq_len >= q_len)
        msk = torch.tril(torch.ones((1, kv_seq_len, kv_seq_len)))
        msk = msk[:, (-q_len):]
        msk = msk.reshape((1, q_len, kv_seq_len))
        invalid_len = kv_seq_len - valid_len
        if invalid_len > 0:
            msk[:,:,:invalid_len] = 0
        out = torch.zeros((1, q_len, kv_seq_len), dtype=torch.float32).masked_fill(msk == 0, 1.0)
        return out

    @staticmethod
    def _build_rope_cache(seq_len: int, n_elem: int, base: int = 10000):
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2) / n_elem))
        seq_idx = torch.arange(seq_len)
        idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
        cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)
        sin[:, :(n_elem // 2)] *= -1
        return cos, sin

    @torch.inference_mode()
    def generate(self, context_ids, max_new_tokens, eos_token_id, pad_token_id, do_sample, temperature=0.5):
        # context_ids: torch.LongTensor, (1 x context length)
        # max_length: context length + maximum generated length
        # eos_token_id: stop criterion
        context_ids = context_ids.squeeze(0)
        context_len = len(context_ids)
        max_position_embeddings = self.config.block_size
        max_length = context_len + max_new_tokens
        assert (context_len < max_length and max_length <= max_position_embeddings)
        device = next(self.buffers()).device

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        #############################################################################
        ## context encoding
        causal_mask = SimModel._make_causal_mask(max_position_embeddings, max_position_embeddings, max_position_embeddings)
        attention_mask = self.config.neg_inf * causal_mask
        position_ids = torch.arange(0, max_position_embeddings, dtype=torch.long)
        input_ids = torch.cat(
            [
                context_ids,
                torch.zeros((max_position_embeddings - context_len), dtype=torch.long, device=context_ids.device),
            ]
        )

        logits, k_cache, v_cache = self(
            input_ids.to(device), 
            attention_mask.to(device), 
            position_ids.to(device),
        )
        k_cache = k_cache[:,:,:-1]
        v_cache = v_cache[:,:,:-1]
        #############################################################################
        next_token_logits = logits[context_len - 1, :]
        output_ids = context_ids.tolist()
        for i in range(max_length - context_len):
            if do_sample:
                next_token = torch.multinomial(torch.softmax(next_token_logits/temperature, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)
            output_ids.append(next_token.item())
            if next_token.item() in eos_token_id:
                break
            attention_mask[:, context_len+i, context_len+i] = self.config.neg_inf
            attention_mask[:, context_len+i, -1] = 0.0
            with torch.no_grad():
                next_token_logits, current_k_cache, current_v_cache = self(
                    next_token, 
                    attention_mask[:, context_len+i].unsqueeze(1).to(device), 
                    position_ids[context_len+i].to(device),
                    k_cache=k_cache,
                    v_cache=v_cache,
                )
            next_token_logits = next_token_logits.squeeze(0)
            if context_len+i+1 < max_position_embeddings:
                k_cache[:,:,context_len+i] = current_k_cache.squeeze(2)
                v_cache[:,:,context_len+i] = current_v_cache.squeeze(2)
        gc.collect()
        torch.cuda.empty_cache()
        return torch.tensor(output_ids, dtype=torch.long).unsqueeze(0)    


class SimMLP(nn.Module):
    def __init__(self, config: SimConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.mlp_bias)
        self.w3 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.mlp_bias)
        if config.impl_sym_pch_as_slinear:
            self.w2 = SLinear(config.intermediate_size, config.n_embd, bias=config.mlp_bias, for_aimet=config.for_aimet)
        else:
            self.w2 = nn.Linear(config.intermediate_size, config.n_embd, bias=config.mlp_bias)
        if config.act_fn.lower() == 'silu':
            self.act = nn.SiLU() 
        elif config.act_fn.lower() == 'gelu':
            self.act = nn.GELU()
        else:
            raise NotImplementedError
        self.mul = torch.mul if config.for_aimet else ElementwiseMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w1(x)
        y = self.act(y)
        z = self.w3(x)
        w = self.mul(y, z)
        x = self.w2(w)
        return x


class SimAttention(nn.Module):
    def __init__(self, config: SimConfig) -> None:
        super().__init__()

        # override the head dim
        config.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_head    * config.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attention_bias)

        self.add_mask = torch.add if config.for_aimet else ElementwiseAdd()
        self.softmax = nn.Softmax(dim=-1)

        self.qk_bmm = torch.matmul if config.for_aimet else FMatMul()
        self.pv_bmm = torch.matmul if config.for_aimet else FMatMul()
        
        self.apply_rope1 = FROPE(config)
        self.apply_rope2 = FROPE(config)

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.seq_len = config.block_size
        self.use_sha = config.use_sha

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, k_cache: Optional[torch.Tensor] = None, v_cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        T, C = x.size()  # sequence length, embedding dimensionality (n_embd)

        q  = self.q_proj(x)
        k_ = self.k_proj(x)
        v_ = self.v_proj(x)

        q  = q.reshape((T, self.n_head,    self.head_dim))
        k_ = k_.reshape((T, self.n_kv_head, self.head_dim))
        v_ = v_.reshape((T, self.n_kv_head, self.head_dim))

        q  = q.transpose(0, 1)  # (nh, T, hs)
        k_ = k_.transpose(0, 1)  # (nh, T, hs)
        v_ = v_.transpose(0, 1)  # (nh, T, hs)

        q = self.apply_rope1(q, cos, sin)
        k_ = self.apply_rope2(k_, cos, sin)

        out_k, out_v = k_, v_
        if k_cache is not None:
            k, v = torch.cat([k_cache, k_], dim=1), torch.cat([v_cache, v_], dim=1)
        else:
            k, v = k_, v_

        new_T = k.size(1)
        if self.n_kv_head < self.n_head:
            assert(self.n_head % self.n_kv_head == 0)
            k = k[:, None].expand(self.n_kv_head, self.n_head // self.n_kv_head, new_T, self.head_dim).contiguous().reshape((self.n_head, -1, self.head_dim)).contiguous()
            v = v[:, None].expand(self.n_kv_head, self.n_head // self.n_kv_head, new_T, self.head_dim).contiguous().reshape((self.n_head, -1, self.head_dim)).contiguous()
        
        # #####################################################################################################################################
        # att = []
        # for i in range(self.n_head):
        #     att.append(self.qk_bmm(q[i], k[i]))
        # # att = torch.stack(att, dim=0)
        # return att[1]
        # #####################################################################################################################################


        # causal self-attention; Self-attend: (nh, T, hs) x (nh, hs, T) -> (nh, T, T)
        # we fused the scaling factor to q_proj
        # if use_kv_cache:
        #     att = torch.matmul(k, q.reshape((self.n_head, self.head_dim, T)))
        #     att = att.transpose(1, 2) # (nh, T, new_T)
        # else:
        #     att = self.qk_bmm(q, k.transpose(1, 2)) # / math.sqrt(self.head_dim)
        # att = self.softmax(att, mask)
        # bm = mask.expand(self.n_head, self.seq_len, self.seq_len).contiguous()
        # att = att.masked_fill(mask == 0, -65536)
        # att = self.qk_bmm(k, q.transpose(1, 2).reshape((self.n_head, self.head_dim, -1)))
        # att = att.transpose(1, 2) # (nh, T, new_T)
        if self.use_sha:
            k_transposed = k.transpose(1, 2)
            att = torch.stack([torch.matmul(q[head_ind], k_transposed[head_ind]) for head_ind in range(self.n_head)], 0)
        else:
            att = self.qk_bmm(q, k.transpose(1, 2)) # / math.sqrt(self.head_dim)
        att = self.add_mask(att, mask)
        att = self.softmax(att)
        y = self.pv_bmm(att, v)
        #  y = att @ v # (nh, T, T) x (nh, T, hs) -> (nh, T, hs)

        # # efficient attention using Flash Attention CUDA kernels
        # y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, :, :T, :T], dropout_p=0.0)

        y = y.transpose(0, 1).contiguous().reshape((T, C))  # re-assemble all head outputs side by side

        # output projection
        y = self.o_proj(y)

        return y, out_k, out_v


class SimBlock(nn.Module):
    def __init__(self, config: SimConfig) -> None:
        super().__init__()
        if config.norm_layer.lower() == "rmsnorm":
            self.input_layernorm = FRMSNorm(config)
            self.post_attention_layernorm = FRMSNorm(config)
        else:
            raise NotImplementedError
        
        self.self_attn = SimAttention(config)
        self.mlp = SimMLP(config)
        self.add1 = torch.add if config.for_aimet else ElementwiseAdd()
        self.add2 = torch.add if config.for_aimet else ElementwiseAdd()

    def forward(self, 
            x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, k_cache: Optional[torch.Tensor] = None, v_cache: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:

        y = self.input_layernorm(x)
        y, k, v = self.self_attn(y, mask, cos, sin, k_cache, v_cache)
        x = self.add1(x, y)
        y = self.post_attention_layernorm(x)
        y = self.mlp(y)
        x = self.add2(x, y)
        return x, k, v


class Sim_QNN(nn.Module):
    def __init__(self, config: SimConfig, n_layers) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(SimBlock(config) for _ in range(n_layers))
        if config.norm_layer.lower() == "rmsnorm":
            self.norm   = FRMSNorm(config)
        else:
            raise NotImplementedError
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.mlp_bias)

    def forward(self, 
            x: torch.Tensor, attention_mask: torch.Tensor, 
            cos: torch.Tensor, sin: torch.Tensor,
            k_cache: Optional[torch.Tensor] = None,
            v_cache: Optional[torch.Tensor] = None,
        ):
        out_k, out_v = [], []
        for i, block in enumerate(self.layers):
            if k_cache is not None:
                # QNN does not have good supports for Gather, so lets use Slice and Squeeze
                x, k, v = block(x, attention_mask, cos, sin, k_cache[i:(i+1), ...].squeeze(0), v_cache[i:(i+1), ...].squeeze(0))
                # x, k, v = block(x, attention_mask, cos, sin, k_cache[i], v_cache[i]) 
            else:
                x, k, v = block(x, attention_mask, cos, sin) # (t, n_embd)
            out_k.append(k)
            out_v.append(v)
        x = self.norm(x)
        logits = self.lm_head(x)  # (b, t, vocab_size)
        out_k = torch.stack(out_k, 0)
        out_v = torch.stack(out_v, 0)
        return logits, out_k, out_v

    @staticmethod
    def from_sim(module, n_layers):
        out = Sim_QNN(module.config, n_layers)
        for i in range(n_layers):
            out.layers[i] = deepcopy(module.layers[i])
        out.norm = deepcopy(module.norm)
        out.lm_head = deepcopy(module.lm_head)
        return out


class Sim_Body(nn.Module):
    def __init__(self, config: SimConfig, n_layers) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(SimBlock(config) for _ in range(n_layers))

    def forward(self, 
            x: torch.Tensor, attention_mask: torch.Tensor, 
            cos: torch.Tensor, sin: torch.Tensor,
            k_cache: Optional[torch.Tensor] = None,
            v_cache: Optional[torch.Tensor] = None,
        ):
        out_k, out_v = [], []
        for i, block in enumerate(self.layers):
            if k_cache is not None:
                # QNN does not have good supports for Gather, so lets use Slice and Squeeze
                x, k, v = block(x, attention_mask, cos, sin, k_cache[i:(i+1), ...].squeeze(0), v_cache[i:(i+1), ...].squeeze(0))
                # x, k, v = block(x, attention_mask, cos, sin, k_cache[i], v_cache[i]) 
            else:
                x, k, v = block(x, attention_mask, cos, sin) # (t, n_embd)
            out_k.append(k)
            out_v.append(v)
        return x, out_k, out_v

    @staticmethod
    def from_sim(module, n_layers):
        out = Sim_Body(module.config, n_layers)
        for i in range(n_layers):
            out.layers[i] = deepcopy(module.layers[i])
        return out


class Sim_Head(nn.Module):
    def __init__(self, config: SimConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)

        # rope
        cos, sin = SimModel._build_rope_cache(config.block_size, n_elem=config.n_embd // config.n_head)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, x: torch.LongTensor, attention_mask: torch.Tensor, position_ids: torch.LongTensor):
        feats = self.embed_tokens(x).view(-1, self.config.n_embd)
        cos, sin = self.cos[position_ids].view(-1, self.config.head_dim), self.sin[position_ids].view(-1, self.config.head_dim)
        return feats, attention_mask, cos, sin

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(SimConfig.from_name(name))

    @staticmethod
    def from_sim(module):
        out = Sim_Head(module.config)
        out.embed_tokens = deepcopy(module.embed_tokens)
        return out