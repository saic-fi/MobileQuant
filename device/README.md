```
CUDA_VISIBLE_DEVICES=0 python device/convert_sim.py --hf_path checkpoints/quantized/llama-1.1b-w4a8

CUDA_VISIBLE_DEVICES=0 python eval/harness_eval.py --hf_path checkpoints/quantized/llama-1.1b-w4a8 --mode sim

CUDA_VISIBLE_DEVICES=0 python device/calibrate.py --hf_path checkpoints/quantized/llama-1.1b-w4a8 --per_channel --use_conv --weight_bitwidth 4 --act_dict_path checkpoints/quantized/llama-1.1b-w4a8/act_dict.json

CUDA_VISIBLE_DEVICES=0 python device/harness_aimet_ctx.py --ctx_encoding results/sim_llama-1.1b-w4a8_calibration/ctx/model_ctx_torch_override.encodings --hf_path checkpoints/quantized/llama-1.1b-w4a8 --per_channel --use_conv --weight_bitwidth 4

CUDA_VISIBLE_DEVICES=0 python device/export.py --hf_path checkpoints/quantized/llama-1.1b-w4a8 --quant_encoding results/sim_llama-1.1b-w4a8_calibration/gen/model_gen_transfered.encodings --kv_cache --per_channel --use_conv --quant_config 4 8 32 --kv_encoding results/sim_llama-1.1b-w4a8_calibration/gen/model_gen_kv_cache.encodings

CUDA_VISIBLE_DEVICES=0 python device/export.py --hf_path checkpoints/quantized/llama-1.1b-w4a8 --quant_encoding results/sim_llama-1.1b-w4a8_calibration/ctx/model_ctx_torch_override.encodings --per_channel --quant_config 4 8 32

```

```

CUDA_VISIBLE_DEVICES=0 python device/convert_sim.py --hf_path checkpoints/quantized/llama-1.1b-w8a8

CUDA_VISIBLE_DEVICES=0 python eval/harness_eval.py --hf_path checkpoints/quantized/llama-1.1b-w8a8 --mode sim

CUDA_VISIBLE_DEVICES=0 python device/calibrate.py --hf_path checkpoints/quantized/llama-1.1b-w8a8 --act_dict_path checkpoints/quantized/llama-1.1b-w8a8/act_dict.json

CUDA_VISIBLE_DEVICES=0 python device/harness_aimet_ctx.py --ctx_encoding results/sim_llama-1.1b-w8a8_calibration/ctx/model_ctx_torch_override.encodings --hf_path checkpoints/quantized/llama-1.1b-w8a8

CUDA_VISIBLE_DEVICES=0 python device/export.py --hf_path checkpoints/quantized/llama-1.1b-w8a8 --quant_encoding results/sim_llama-1.1b-w8a8_calibration/gen/model_gen_transfered.encodings --kv_cache --quant_config 8 8 32 --kv_encoding results/sim_llama-1.1b-w8a8_calibration/gen/model_gen_kv_cache.encodings

```

```

CUDA_VISIBLE_DEVICES=0 python device/convert_sim.py --hf_path checkpoints/quantized/gemma-2b-w4a8

CUDA_VISIBLE_DEVICES=0 python eval/harness_eval.py --hf_path checkpoints/quantized/gemma-2b-w4a8 --mode sim

CUDA_VISIBLE_DEVICES=0 python device/calibrate.py --hf_path checkpoints/quantized/gemma-2b-w4a8 --per_channel --use_conv --weight_bitwidth 4 --act_dict_path checkpoints/quantized/gemma-2b-w4a8/act_dict.json

CUDA_VISIBLE_DEVICES=0 python device/harness_aimet_ctx.py --ctx_encoding results/sim_gemma-2b-w4a8_calibration/ctx/model_ctx_torch_override.encodings --hf_path checkpoints/quantized/gemma-2b-w4a8 --per_channel --use_conv --weight_bitwidth 4

CUDA_VISIBLE_DEVICES=0 python device/export.py --hf_path checkpoints/quantized/gemma-2b-w4a8 --quant_encoding results/sim_gemma-2b-w4a8_calibration/gen/model_gen_transfered.encodings --kv_encoding results/sim_gemma-2b_calibration/gen/model_gen_kv_cache.encodings --kv_cache --per_channel --use_conv --quant_config 4 8 32
```


```

CUDA_VISIBLE_DEVICES=0 python device/convert_sim.py --hf_path checkpoints/quantized/gemma-2b-w8a8

CUDA_VISIBLE_DEVICES=0 python eval/harness_eval.py --hf_path checkpoints/quantized/gemma-2b-w8a8 --mode sim

CUDA_VISIBLE_DEVICES=0 python device/calibrate.py --hf_path checkpoints/quantized/gemma-2b-w8a8 --act_dict_path checkpoints/quantized/gemma-2b-w8a8/act_dict.json

CUDA_VISIBLE_DEVICES=0 python device/harness_aimet_ctx.py --ctx_encoding results/sim_gemma-2b-w8a8_calibration/ctx/model_ctx_torch_override.encodings --hf_path checkpoints/quantized/gemma-2b-w8a8

CUDA_VISIBLE_DEVICES=0 python device/export.py --hf_path checkpoints/quantized/gemma-2b-w8a8 --quant_encoding results/sim_gemma-2b-w8a8_calibration/gen/model_gen_transfered.encodings --kv_encoding results/sim_gemma-2b-w8a8_calibration/gen/model_gen_kv_cache.encodings --kv_cache
```