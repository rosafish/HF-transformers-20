quality=high
test_type=mismatched
seed=0
train_size=240
server=uchi # ego or uchi

if [ $server = ego ]; then

    data_path_prefix=/data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/
    save_model_path_prefix=./save_best_models/

elif [ $server = uchi ]; then

    data_path_prefix=~/data/randomness_experiment/
    save_model_path_prefix=/net/scratch/zhouy1/randomness_experiment/edm/

fi

python ./esnli_bert2bert_eval.py\
    -model_dir ${save_model_path_prefix}bert_hans_seed${seed}_train${train_size}_${quality}/best_model/ \
    -eval_data_path ${data_path_prefix}seed${seed}/${test_type}_test3000_${quality}.csv \
    -eval_results_dir ${save_model_path_prefix}bert_hans_seed${seed}_train${train_size}_${quality}/eval_${test_type}_test/ 