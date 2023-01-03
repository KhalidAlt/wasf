

model_id="facebook/m2m100-12B-last-ckpt"
data_name='conceptual_captions'
cache_dir="/home/ubuntu/ccg/cache/"
save_path="/home/ubuntu/ccg/dataset/"


python translate_hf.py  \
    --model_name_or_path $model_id \
    --tokenizer_name_or_path $model_id \
    --dataset $data_name \
    --split "train" \
    --subset 'unlabeled' \
    --column_name "caption" \
    --device "cpu" \
    --save_path $save_path \
