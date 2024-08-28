# [MobileQuant: Mobile-friendly Quantization for On-device Language Models](http://arxiv.org/abs/2408.13933)

## Abstract
Large language models (LLMs) have revolutionized language processing, delivering outstanding results across multiple applications. However, deploying LLMs on edge devices poses several challenges with respect to memory, energy, and compute costs, limiting their widespread use in devices such as mobile phones. A promising solution is to reduce the number of bits used to represent weights and activations. While existing works have found partial success at quantizing LLMs to lower bitwidths, e.g. 4-bit weights, quantizing activations beyond 16 bits often leads to large computational overheads due to poor on-device quantization support, or a considerable accuracy drop. Yet, 8-bit activations are very attractive for on-device deployment as they would enable LLMs to fully exploit mobile-friendly hardware, e.g. Neural Processing Units (NPUs). In this work, we make a first attempt to facilitate the on-device deployment of LLMs using integer-only quantization. We first investigate the limitations of existing quantization methods for on-device deployment, with a special focus on activation quantization. We then address these limitations by introducing a simple post-training quantization method, named \method{}, that extends previous weight equivalent transformation works by jointly optimizing the weight transformation and activation range parameters in an end-to-end manner. MobileQuant demonstrates superior capabilities over existing methods by 1) achieving near-lossless quantization on a wide range of LLM benchmarks, 2) reducing latency and energy consumption by 20\%-50\% compared to current on-device quantization strategies, 3) requiring limited compute budget, 4) being compatible with mobile-friendly compute units, e.g. NPU.

## Install
Please follow the instructions in [INSTALL](INSTALL.md).

## Demo
Please check out the [capp](./capp) folder if you're interested in playing with the on-device LLM demo. This demo requires a mobile phone with a [Snapdragon 8 Gen 3](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-3-mobile-platform) NPU.

## Benchmarking
Please check out the [device](./device) folder if you're interested in benchmarking the on-device LLM models. This requires a mobile phone with a [Snapdragon 8 Gen 3](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-3-mobile-platform) NPU.

## Evaluation
Please check out the [eval](./eval) folder if you'd like to evaluate the pre-quantized models.

## Quantization
This work has been tested on three representative models: [TinyLlaMA-1.1b-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), [gemma-2b](https://huggingface.co/google/gemma-2b), [stablelm-2-1_6b](https://huggingface.co/stabilityai/stablelm-2-1_6b).
The models are chosen as they are small enough to be deployed on mobile devices, and they have different architectures.

### Convert the HF ckpt
We need to first convert the HF checkpoints to our format, which is also in HF format. It is a unified model definition for these models, which is implemented in [hf_model.py](mobilellm/model/hf_model.py). 

```bash
# Convert the HF ckpt
CUDA_VISIBLE_DEVICES=0 python scripts/convert_ckpt.py --checkpoint_dir ${HF_CKPT} --output_dir ${NEW_HF_CKPT}

# You can check if the performance of the new ckpt matches the HF ckpt by running
CUDA_VISIBLE_DEVICES=0 python eval/simple_eval.py --hf_path ${NEW_HF_CKPT}
```

Alternatively, we can download the pre-converted models:
```bash
# TinyLlaMA-1.1b-Chat-v1.0
git lfs install
git clone https://huggingface.co/fwtan/llama-1.1b-converted

# stablelm-2-1_6b
git lfs install
git clone https://huggingface.co/fwtan/stablelm-2-1.6b-converted

# gemma-2B
git lfs install
git clone https://huggingface.co/fwtan/gemma-2b-converted
```

### Quantization scripts

We include quantization scripts of both the baselines methods, i.e. SmoothQuant, OmniQuant, and our approach in the [experiments](./experiments) folder.
As OmniQuant and MobileQuant require proper initialization from SmoothQuant, please run the SmoothQuant scripts before running OmniQuant or MobileQuant.







