pretrained_model=esnli # esnli or bert

quality=$1
seed=$2
data_type=$3 # mvmt or mvmist or misvmt or misvmist or dev
train_size=$4
dev_size=$5

server=uchi # ego or uchi

if [ $server = ego ]; then

    data_path_prefix=/data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/
    save_model_path_prefix=./save_best_models/

elif [ $server = uchi ]; then

    data_path_prefix=~/data/randomness_experiment/
    save_model_path_prefix=/net/scratch/zhouy1/randomness_experiment/edm/

fi

if [[ $data_type = mvmt || $data_type = misvmt ]]; then

    test_size=12000
    python ./esnli_bert2bert_eval.py\
    -model_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_train${train_size}_${quality}/best_model/ \
    -eval_data_path ${data_path_prefix}seed${seed}/${data_type}_test${test_size}_${quality}.csv \
    -eval_results_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_train${train_size}_${quality}/eval_${data_type}_test/

elif [[ $data_type = mvmist || $data_type = misvmist ]]; then

    test_size=3000
    python ./esnli_bert2bert_eval.py\
    -model_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_train${train_size}_${quality}/best_model/ \
    -eval_data_path ${data_path_prefix}seed${seed}/${data_type}_test${test_size}_${quality}.csv \
    -eval_results_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_train${train_size}_${quality}/eval_${data_type}_test/

elif [ $data_type = dev ]; then

    python ./esnli_bert2bert_eval.py\
    -model_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_train${train_size}_${quality}/best_model/ \
    -eval_data_path ${data_path_prefix}seed${seed}/${data_type}${dev_size}_${quality}.csv \
    -eval_results_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_train${train_size}_${quality}/eval_${data_type}/
    
fi

 