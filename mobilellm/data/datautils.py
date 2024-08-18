import numpy as np
import torch, random
from tqdm import tqdm
from datasets import load_dataset


def get_pile(nsamples, seed, seqlen, tokenizer):
    print("Getting pile")
    traindata = load_dataset("json", data_files='data/pile/val.jsonl.zst', split=f"train[:{nsamples}]") 
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    random.seed(seed)
    trainloader = []
    for _ in tqdm(range(nsamples)):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, None


def get_wikitext(nsamples, seed, seqlen, tokenizer):
    print("Getting wikitext")
    traindata = load_dataset('wikitext', 'wikitext-103-raw-v1', split=f"train")
    testdata = load_dataset('wikitext', 'wikitext-103-raw-v1', split=f"test")
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    print("Getting wikitext2")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split=f"train")
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split=f"test")
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_loaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    
    if 'wikitext' in name:
        return get_wikitext(nsamples, seed, seqlen, tokenizer)

    if 'pile' in name:
        return get_pile(nsamples, seed, seqlen,  tokenizer)

    raise NotImplementedError
