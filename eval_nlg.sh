export CUDA_VISIBLE_DEVICES=1

WEIGHT_DIR="../defeasible/weights_nlg_only"

########### Defeasible - ATOMIC
python eval_nlg.py \
    --model_name gpt2-xl \
    --data_name snli \
    --input_mode phu \
    --only_weakener \
    --output_dir $WEIGHT_DIR \
    --eval_step -1 \
    --train_batch_size 64 \
    --eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lr 1e-5 \
    --max_seq_length 128 \
    --seed 10
