# original implementation
CUDA_VISIBLE_DEVICES=0 python ptq/smoothquant.py --hf_path checkpoints/baselines/llama-1.1b --original_smoothquant --alpha 0.5
CUDA_VISIBLE_DEVICES=0 python ptq/generate_act_range.py --hf_path checkpoints/baselines/llama-1.1b
CUDA_VISIBLE_DEVICES=0 python ptq/generate_qcfg.py --hf_path checkpoints/baselines/llama-1.1b --use_16bit_softmax_input --use_16bit_softmax_output
CUDA_VISIBLE_DEVICES=0 python eval/harness_eval.py --hf_path checkpoints/baselines/llama-1.1b --mode custom --tasks wikitext --max_length 2048