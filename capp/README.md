<div align="center">

## LLM on-device demo

</div>

We provide a simple demo to run the quantized LLMs on device. Currently, the app supports running W4A8 and W8A8 LlaMA models on an Android phone with a [Snapdragon 8 Gen 3](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-3-mobile-platform)  NPU.

The demo was initially developed by [Lukasz Dudziak](https://github.com/ldudziak) for stable-diffusion. [Shell Xu Hu](https://github.com/hushell) further adapted the code for prompt-encoding. 
[Fuwen Tan](https://github.com/fwtan) finalized the code by implementing the KV cache and auto-regressive generation.

## Pre-compiled app and models

You're welcome to try out the precompiled app and models directly on an Android phone with a [Snapdragon 8 Gen 3](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-3-mobile-platform) HTP, e.g. Samsung Galaxy S24, Xiaomi 14, etc.

- get the quantized on-device models

```
# W8A8
git lfs install
git clone https://huggingface.co/fwtan/llama-1.1b-mobilequant-w8a8-s1024-e60-8gen3

# W4A8
git lfs install
git clone https://huggingface.co/fwtan/llama-1.1b-mobilequant-w4a8-s1024-e60-sym-8gen3
```

- push the precompile app to the phone

```
git clone https://huggingface.co/fwtan/llm_8gen3_demo
adb push llm_8gen3_demo /data/local/tmp/
```

- push the quantized models to the phone

```
# W8A8
adb push llama-1.1b-mobilequant-w8a8-s1024-e60-8gen3 /data/local/tmp/llm_8gen3_demo

# W4A8
adb push llama-1.1b-mobilequant-w4a8-s1024-e60-sym-8gen3 /data/local/tmp/llm_8gen3_demo
```

- run the demo

```
# W8A8
adb shell "cd /data/local/tmp/llm_8gen3_demo && LD_LIBRARY_PATH=. ./simple_app llama-1.1b-mobilequant-w8a8-s1024-e60-8gen3"

# W4A8
adb shell "cd /data/local/tmp/llm_8gen3_demo && LD_LIBRARY_PATH=. ./simple_app llama-1.1b-mobilequant-w4a8-s1024-e60-sym-8gen3"
```

## Build the demo yourself

### :panda_face: Installation
The code requires `clang-16`, `QNN` and `Android NDK`. To install `clang-16` using `apt`:
```
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
```
Add this line to `/etc/apt/sources.list`
```
deb http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main
```
and run
```
sudo apt update
sudo apt install clang-16 lldb-16 lld-16
```

The code also depends on [QNN 2.22](https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.22.6.240515.zip?query=aiesdk) and Android-NDK. Make sure you have these two env variables defined:
```
export QNN_SDK_ROOT=/path/to/qnn-2.22
export ANDROID_NDK_ROOT=/path/to/android_sdk/ndk-bundle
```

### :hammer: Compilation 
```
make aarch64-android
```
This should populate the `bin/aarch64-android` folder with the required files in the demo:
```
libc++_shared.so*  libllmod.so*  libQnnHtp.so*	libQnnHtpV75Skel.so*  libQnnHtpV75Stub.so*  libQnnSystem.so*  simple_app*
```

### :running: Preparing `meta.bin`, `tokenizer.bin`, and `qnn_model.bin`
We need to prepare three extra files: `meta.bin` that stores the embedding layer, `tokenizer.bin` that stores the LlaMA tokenizer, `qnn_model.bin` that stores the model graph. 

- prepare `meta.bin`

```
python scripts/export_bin.py meta.bin --hf /path/to/TinyLlama-1.1B-Chat-v1.0
```

- prepare `tokenizer.bin`

```
python scripts/tokenizer.py -t /path/to/TinyLlama-1.1B-Chat-v1.0/tokenizer.model -c /path/to/TinyLlama-1.1B-Chat-v1.0/tokenizer_config.json
```

- prepare `qnn_model.bin`

  Please checkout the profiling section ([device/README.md](../device/README.md))

