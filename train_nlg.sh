export WANDB_PROJECT=defeasible_gen2

########### Defeasible -SNLI

# Train with only weakener samples
# 1 Epoch, 64 batch
# Eval every 80 steps
python train_final.py \
    --model_name gpt2-xl \
    --data_name snli \
    --input_mode phu \
    --only_weakener \
    --output_dir ./weights_nlg_only \
    --epochs 1 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr 1e-5 \
    --max_seq_length 128 \
    --eval_steps 80 \
    --seed 10 \
    --parallelize \
    --fp16 

########### Defeasible - ATOMIC

# Train with only weakener samples
# 1 Epoch, 64 batch
# Eval every 40 steps

python train_final.py \
    --model_name gpt2-xl \
    --data_name atomic \
    --input_mode phu \
    --only_weakener \
    --output_dir ./weights_nlg_only \
    --epochs 1 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr 1e-5 \
    --max_seq_length 128 \
    --eval_steps 40 \
    --seed 10 \
    --parallelize \
    --fp16 