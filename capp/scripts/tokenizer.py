# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os, struct, argparse
from typing import List
from sentencepiece import SentencePieceProcessor
from mobilellm.utils.io import json_load


TOKENIZER_MODEL = "tokenizer.model"
TOKENIZER_CONFIG = "tokenizer_config.json"


class Tokenizer:
    def __init__(self, tokenizer_model=None, tokenizer_config=None):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        config_path = tokenizer_config if tokenizer_config else TOKENIZER_CONFIG
        
        assert os.path.isfile(model_path), model_path
        assert os.path.isfile(config_path), config_path

        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path
        

        


        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        # print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

        self.special_i2t = {self.bos_id: "<s>", self.eos_id: "</s>"}
        for ind, info in json_load(tokenizer_config)["added_tokens_decoder"].items():
            self.special_i2t[int(ind)] = info["content"]
        # self.special_t2i = {}
        # for ind, tok in self.special_i2t.items():
        #     self.special_t2i[tok] = ind

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def export(self):

        # get all the tokens (postprocessed) and their scores as floats
        id2pair = {}
        for i in range(self.n_words):
            # decode the token and light postprocessing
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = f'\n{self.special_i2t[self.bos_id]}\n'
                # t = self.special_i2t[self.bos_id]
                # print('bos', t, s)
            elif i == self.eos_id:
                t = f'\n{self.special_i2t[self.eos_id]}\n'
                # t = self.special_i2t[self.eos_id]
                # print('eos', t, s)
            t = t.replace('▁', ' ') # sentencepiece uses this character as whitespace
            b = t.encode('utf-8') # bytes of this token, utf-8 encoded
            id2pair[i] = (b, s)

        for ind, t in self.special_i2t.items():
            assert ind in id2pair
            t = t.replace('▁', ' ') # sentencepiece uses this character as whitespace
            b = t.encode('utf-8') # bytes of this token, utf-8 encoded
            id2pair[ind] = (b, 0.0)

        tokens, scores = [], []
        for i in range(self.n_words):
            b, s = id2pair[i]
            tokens.append(b)
            scores.append(s)
            # print(b, s)


        # record the max token length
        max_token_length = max(len(t) for t in tokens)
        print("vocab size:", len(tokens))

        # write to a binary file
        # the tokenizer.bin file is the same as .model file, but .bin
        tokenizer_bin = self.model_path.replace('.model', '.bin')
        with open(tokenizer_bin, 'wb') as f:
            f.write(struct.pack("I", max_token_length))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer-model", type=str, help="optional path to custom tokenizer ")
    parser.add_argument("-c", "--tokenizer-config", type=str, help="optional path to custom config")
    args = parser.parse_args()

    t = Tokenizer(args.tokenizer_model, args.tokenizer_config)
    t.export()
