cd ../model_train
for lr in 1e-3
do
for seed in 27
do
attn_lowdim=128
ffn_lowdim=384
low_attn=yes
low_ffn=no
echo $attn_lowdim
echo $ffn_lowdim
echo $lr
echo $seed
echo '---------'
torchrun --nproc_per_node 8 \
lpa_train_setting2.py \
    --train_data_path ../datasets/tokenized/train00_256 \
    --valid_data_path ../datasets/tokenized/val_256 \
    --test_data_path ../datasets/tokenized/test_256 \
    --tokenizer_path ../transformers/llama-2/llama-2-7b-hf \
    --block_size 256 \
    --embed_dim 768 \
    --ffn_dim 2048 \
    --num_heads 12 \
    --head_dim 64 \
    --attn_lowdim $attn_lowdim \
    --ffn_lowdim $ffn_lowdim \
    --num_layers 12 \
    --low_attn $low_attn \
    --low_ffn $low_ffn \
    --do_train \
    --do_eval \
    --seed $seed \
    --learning_rate $lr \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --max_steps 31800 \
    --logging_steps 20 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --bf16 True \
    --output_dir ../results \
    --report_to none \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --lr_scheduler_type cosine
done
done