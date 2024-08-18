EPOCHS=20
NSAMPLES=128
CKPT=checkpoints/hfmodels/gemma-2b
OUTPUT_DIR=results/gemma-2b-lrl-w8a8
BATCH_SIZE=1
LET_LR=5e-3
LET_MIN_LR=5e-3
LWC_LR=1e-2
LWC_MIN_LR=5e-3
LRL_LR=1e-6
LRL_MIN_LR=1e-7


CUDA_VISIBLE_DEVICES=0 python ptq/mobilequant.py --lwc --let --lrl --tasks wikitext \
    --act_bitwidth 8 --epochs ${EPOCHS} --nsamples ${NSAMPLES} --dtype float32 \
    --deactive_amp --hf_path ${CKPT} --output_dir ${OUTPUT_DIR} \
    --let_lr ${LET_LR} --let_min_lr ${LET_MIN_LR} \
    --lwc_lr ${LWC_LR} --lwc_min_lr ${LWC_MIN_LR} \
    --lrl_lr ${LRL_LR} --lrl_min_lr ${LRL_MIN_LR} \
    --batch_size ${BATCH_SIZE} --weight_bitwidth 8


CUDA_VISIBLE_DEVICES=0 python eval/harness_eval.py --tasks wikitext \
    --mode custom --hf_path ${OUTPUT_DIR}