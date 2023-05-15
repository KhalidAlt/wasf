

BIGS_MODEL="roberta-base"
tokenizer_name="khalidalt/tokenizer_bpe"
#data_path='khalidalt/hazmah'
data_path='khalidalt/sample'
cache_dir="/media/khalid/data_disk/Cache_for_LM/Roberta"
output_dir="/home/khalid/Documents/github_rep/MyProjects/CodeBase/training/output/roberta/"
logging_dir="/home/khalid/Documents/github_rep/MyProjects/CodeBase/training/logging/roberta/"
ds_config="/home/khalid/Documents/github_rep/MyProjects/CodeBase/training/ds_config.json"
mkdir -p $output_dir
mkdir -p $logging_dir

CUDA_VISIBLE_DEVICES=1 python ./run_mlm_update.py  \
    --model_name_or_path roberta-base \
    --tokenizer_name $tokenizer_name \
    --dataset_name $data_path \
    --cache_dir $cache_dir \
    --output_dir $output_dir \
    --logging_dir $logging_dir \
    --warmup_steps 24_000 \
    --weight_decay 0.01 \
    --report_to "wandb" \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --num_train_epochs 2 \
    --do_eval \
    --overwrite_output_dir \
    --max_seq_length 512 \
    --max_eval_samples 2 \
