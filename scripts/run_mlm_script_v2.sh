BIGS_MODEL="roberta-base"
tokenizer_name="khalidalt/tokenizer_bpe"
data_path='khalidalt/Joud'
config_name='all'
#data_path="xnli"
#config_name="ar"
cache_dir="/fsx/home-khalida/ds_cache"
output_dir="/fsx/home-khalida/model/output/roberta/"
logging_dir="/fsx/home-khalida/logging/roberta/"

mkdir -p $output_dir
mkdir -p $logging_dir

python ./run_mlm_v2.py  \
    --model_name_or_path roberta-base \
    --tokenizer_name $tokenizer_name \
    --dataset_name $data_path \
    --dataset_config_name  $config_name \
    --cache_dir $cache_dir \
    --output_dir $output_dir \
    --logging_dir $logging_dir \
    --warmup_steps 24_000 \
    --weight_decay 0.01 \
    --report_to "wandb" \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --num_train_epochs 45 \
    --do_eval \
    --overwrite_output_dir \
    --max_seq_length 512 \
    --streaming \
