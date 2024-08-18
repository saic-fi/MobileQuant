import torch
import transformers
from tqdm import tqdm
from typing import Dict, List, Literal, Optional

from lm_eval import base, evaluator, tasks
from lm_eval.base import BaseLM

from mobilellm.model.sim_model import SimConfig, SimModel


def print_model_size(model, include_buffers=False, include_trainable_only=False):
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    param_cnt = 0
    for param in model.parameters():
        if (not include_trainable_only) or param.requires_grad:
            param_cnt += param.nelement()
            param_size += param.nelement() * param.element_size()
    buffer_size = 0
    buffer_cnt = 0
    if include_buffers:
        for buffer in model.buffers():
            buffer_cnt += buffer.nelement()
            buffer_size += buffer.nelement() * buffer.element_size()
    cnt_all = (param_cnt + buffer_cnt) / 10**9
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}B/{:.3f}MB'.format(cnt_all, size_all_mb))


class Evaluator:
    def __init__(self, dataset, tokenizer, max_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    @torch.inference_mode()
    def evaluate(self, model):
        model.eval()
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latency = 0

        if not isinstance(model, transformers.PreTrainedModel):
            attention_mask = SimModel._make_causal_mask(self.max_length, self.max_length, self.max_length)
            attention_mask = model.config.neg_inf * attention_mask
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

            if isinstance(model, transformers.PreTrainedModel):
                outputs = model(input_ids)
                logits = outputs.logits
            else:
                outputs = model(input_ids[0], attention_mask=attention_mask, position_ids=position_ids)[0]
                logits = outputs.unsqueeze(0)

            end.record()
            torch.cuda.synchronize()
            latency += start.elapsed_time(end)
            last_token_logits = logits[:, -2-pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()

        acc = hit / total
        latency = latency / len(self.dataset)
        return acc, latency


class LMEvalAdaptor(BaseLM):
    def __init__(self, model_name, model, tokenizer, batch_size=1, max_length=-1):
        super().__init__()

        self.model_name = model_name
        self.model = model
        self.model.eval()
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
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            kwargs = {}
            if not isinstance(self.model, transformers.PreTrainedModel):
                # on-device model without the batch dimension
                inps = inps[:, :self.max_length]
                cur_len = inps.shape[1]
                if cur_len < self.max_length:
                    pad_len = self.max_length - inps.shape[1]
                    inps = torch.nn.functional.pad(inps, (0, pad_len), value=0)

                attention_mask = SimModel._make_causal_mask(self.max_length, self.max_length, self.max_length)
                attention_mask = self.model.config.neg_inf * attention_mask
                attention_mask = attention_mask.to(self.device)
                position_ids = torch.arange(0, self.max_length, dtype=torch.long).to(self.device)
                kwargs = {'attention_mask': attention_mask, 'position_ids': position_ids}

            if isinstance(self.model, transformers.PreTrainedModel):
                out = self.model(inps, **kwargs)["logits"]
            else:
                out = self.model(inps[0], **kwargs)[0][:cur_len].unsqueeze(0)
        return out

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context,
            max_new_tokens=self.max_gen_toks,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False
        )