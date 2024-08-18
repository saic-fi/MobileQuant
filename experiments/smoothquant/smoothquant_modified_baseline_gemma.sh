# our implementation
CUDA_VISIBLE_DEVICES=0 python ptq/smoothquant.py --hf_path checkpoints/baselines/gemma-2b --alpha 0.5
CUDA_VISIBLE_DEVICES=0 python ptq/generate_act_range.py --hf_path checkpoints/baselines/gemma-2b
CUDA_VISIBLE_DEVICES=0 python ptq/generate_qcfg.py --hf_path checkpoints/baselines/gemma-2b --use_16bit_softmax_input --use_16bit_softmax_output 
CUDA_VISIBLE_DEVICES=0 python eval/harness_eval.py --hf_path checkpoints/baselines/gemma-2b --mode custom --tasks wikitext --max_length 2048