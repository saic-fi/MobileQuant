<div align="center">

## LLM on-device profiling

</div>

We provide code to profile the quantized model on an Android phone with a [Snapdragon 8 Gen 3](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-3-mobile-platform) NPU.
Please install the necessary packages, e.g. [AIMET](https://github.com/fwtan/aimet/tree/user/fuwen.tan/main), [QNN SDK](https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.22.6.240515.zip?query=aiesdk), following [INSTALL.md](../INSTALL.md)

### :panda_face: Extra requirement for Gemma 2B
If you're playing with LlaMA 1.1B, please ignore this part as it is completely unnecessary.

For models larger than 2B, e.g. [Gemma 2B](https://huggingface.co/google/gemma-2b), we need a `gcc` that uses 64-bit memory addresses.
Our solution is to compile our own `gcc`😢.

- get the `gcc`
```
git clone https://github.com/fwtan/gcc.git
git checkout gcc-12/qnn
```
- compile the `gcc`
```
./contrib/download_prerequisites
cd ..
mkdir objdir
cd objdir
$PWD/../gcc/configure --prefix=$HOME/gcc-qnn --enable-languages=c,c++,fortran,go
make
make install
```

Please do let us know if you have any simpler solutions🙂!

### :running: Profiling

Please `adb connect` to an Android phone with a [Snapdragon 8 Gen 3](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-3-mobile-platform) HTP before running the profiling.

#### Benchmarking W4A8 LlaMA-1.1B 

- get the HF model

```
git clone https://huggingface.co/fwtan/llama-1.1b-mobilequant-w4a8-s1024-e60-sym-hf
export HF_PATH=llama-1.1b-mobilequant-w4a8-s1024-e60-sym-hf
export HF_NAME=llama-1.1b-mobilequant-w4a8-s1024-e60-sym-hf
```

- convert the HF model to a simpler format (`sim` model) that can be parsed by [AIMET](https://github.com/fwtan/aimet/tree/user/fuwen.tan/main). This step is only necessary if you'd like to use your own pretrained model.

```
# Not necessry if you're using the provided model
# CUDA_VISIBLE_DEVICES=0 python device/convert_sim.py --hf_path ${HF_PATH}
```

- export the ONNX files and the quantization encodings. Note that `${HF_PATH}/act_dict.json` will be used to override the quantization encodings from the AIMET calibration.

```
CUDA_VISIBLE_DEVICES=0 python device/calibrate.py --hf_path ${HF_PATH} --per_channel \
    --use_conv --weight_bitwidth 4 --act_dict_path ${HF_PATH}/act_dict.json
```

- export the on-device model and run the profiling

```
CUDA_VISIBLE_DEVICES=0 python device/export.py --hf_path ${HF_PATH} \
    --kv_cache --per_channel --use_conv --quant_config 4 8 32 \
    --quant_encoding results/sim_${HF_NAME}_calibration/gen/model_gen_transfered.encodings \
    --kv_encoding results/sim_${HF_NAME}_calibration/gen/model_gen_kv_cache.encodings
```

The script will report the average latency (ms), and save the profiling data `qnn_profiling.csv` and model `qnn_model.bin` in `sim_${HF_NAME}_qnn_with_kv/device`.
The `qnn_model.bin` file can then be used for the provided demo, as discussed in [capp/README.md](../capp/README.md)


#### Benchmarking W8A8 LlaMA-1.1B 

- get the HF model

```
git clone https://huggingface.co/fwtan/llama-1.1b-mobilequant-w8a8-s1024-e60-hf
export HF_PATH=llama-1.1b-mobilequant-w8a8-s1024-e60-hf
export HF_NAME=llama-1.1b-mobilequant-w8a8-s1024-e60-hf
```

- convert the HF model if necessary

```
# Not necessry if you're using the provided model
# CUDA_VISIBLE_DEVICES=0 python device/convert_sim.py --hf_path ${HF_PATH}
```

The other commands are very similar to W4A8 except we don't use the flags `--per_channel`, `--use_conv`

- export the ONNX files and the quantization encodings.

```
CUDA_VISIBLE_DEVICES=0 python device/calibrate.py --hf_path ${HF_PATH} \
    --weight_bitwidth 8 --act_dict_path ${HF_PATH}/act_dict.json
```

- export the on-device model and run the profiling

```
CUDA_VISIBLE_DEVICES=0 python device/export.py --hf_path ${HF_PATH} \
    --kv_cache --quant_config 8 8 32 \
    --quant_encoding results/sim_${HF_NAME}_calibration/gen/model_gen_transfered.encodings \
    --kv_encoding results/sim_${HF_NAME}_calibration/gen/model_gen_kv_cache.encodings
```


### Benchmarking Gemma 2B

The commands are the same as LlaMA 1.1B, as long as `${HF_PATH}` and `${HF_NAME}` are set properly.
