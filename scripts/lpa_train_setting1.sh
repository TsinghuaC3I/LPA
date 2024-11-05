cd ../model_train
for lr in 5e-4
do
for seed in 27
do
attn_lowdim=320
ffn_lowdim=768
low_attn=yes
low_ffn=no
echo $attn_lowdim
echo $ffn_lowdim
echo $lr
echo $seed
echo '---------'
torchrun --nproc_per_node 8 \
lpa_train_setting1.py \
    --train_data_file ../datasets/wikitext-103-raw/wiki_train.txt \
    --valid_data_file ../datasets/wikitext-103-raw/wiki_valid.txt \
    --test_data_file ../datasets/wikitext-103-raw/wiki_test.txt \
    --tokenizer_path ../transformers/llama-2/llama-2-7b-hf \
    --block_size 1024 \
    --embed_dim 1024 \
    --ffn_dim 4096 \
    --num_heads 8 \
    --head_dim 128 \
    --attn_lowdim $attn_lowdim \
    --ffn_lowdim $ffn_lowdim \
    --num_layers 24 \
    --dropout 0.1 \
    --low_attn $low_attn \
    --low_ffn $low_ffn \
    --do_train \
    --do_eval \
    --seed $seed \
    --learning_rate $lr \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --num_train_epochs 8 \
    --logging_steps 20 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 1000 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --lr_scheduler_type cosine \
    --output_dir ../results \
    --save_strategy no \
    --report_to none
done
done