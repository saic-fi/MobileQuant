MODEL=gemma-2b
CKPT=checkpoints/hfmodels/${MODEL}
BATCH_SIZE=1
LET_LR=1e-3
LET_MIN_LR=1e-3
LWC_LR=5e-3
LWC_MIN_LR=5e-3
LRL_LR=1e-6
LRL_MIN_LR=1e-7

EPOCHS=60
NSAMPLES=1024
OUTPUT_DIR=results/${MODEL}-e2e-w8a8-s${NSAMPLES}-e${EPOCHS}

python ptq/mobilequant.py --lwc --let --lrl \
    --act_bitwidth 8 --epochs ${EPOCHS} --nsamples ${NSAMPLES} --dtype float32 \
    --deactive_amp --hf_path ${CKPT} --output_dir ${OUTPUT_DIR} \
    --let_lr ${LET_LR} --let_min_lr ${LET_MIN_LR} \
    --lwc_lr ${LWC_LR} --lwc_min_lr ${LWC_MIN_LR} \
    --lrl_lr ${LRL_LR} --lrl_min_lr ${LRL_MIN_LR} \
    --batch_size ${BATCH_SIZE} --mode e2e --weight_bitwidth 8


CUDA_VISIBLE_DEVICES=0 python eval/harness_eval.py --tasks wikitext \
    --mode custom --hf_path ${OUTPUT_DIR} --output_dir ${OUTPUT_DIR}