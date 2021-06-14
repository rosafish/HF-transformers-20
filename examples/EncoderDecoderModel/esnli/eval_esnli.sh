pretrained_model=esnli # esnli or bert

quality=$1
seed=$2
partition=$3
data_type=$4 # ivit or ivot or ovit or ovot or dev
train_size=$5
dev_size=$6
data_dir_name=$7
test_size=300

server=uchi # ego or uchi

if [ $server = ego ]; then

    data_path_prefix=/data/rosa/hans-forked/auto/${data_dir_name}/
    save_model_path_prefix=./save_best_models/

elif [ $server = uchi ]; then

    data_path_prefix=/net/scratch/zhouy1/data/${data_dir_name}/
    save_model_path_prefix=/net/scratch/zhouy1/randomness_experiment/${data_dir_name}/edm/

fi

if [ $data_type = ivit ] || [ $data_type = ovit ]; then

    python ./esnli_bert2bert_eval.py\
    -model_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_partition${partition}_train${train_size}_${quality}/best_model/ \
    -eval_data_path ${data_path_prefix}seed${seed}/partition${partition}/test_${data_type}_${test_size}_${quality}.csv \
    -eval_results_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_partition${partition}_train${train_size}_${quality}/eval_${data_type}_test/

elif [ $data_type = ivot ] || [ $data_type = ovot ]; then

    python ./esnli_bert2bert_eval.py\
    -model_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_partition${partition}_train${train_size}_${quality}/best_model/ \
    -eval_data_path ${data_path_prefix}seed${seed}/partition${partition}/test_${data_type}_${test_size}_${quality}.csv \
    -eval_results_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_partition${partition}_train${train_size}_${quality}/eval_${data_type}_test/

elif [ $data_type = dev ]; then

    python ./esnli_bert2bert_eval.py\
    -model_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_partition${partition}_train${train_size}_${quality}/best_model/ \
    -eval_data_path ${data_path_prefix}seed${seed}/partition${partition}/${data_type}_${dev_size}_${quality}.csv \
    -eval_results_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_partition${partition}_train${train_size}_${quality}/eval_${data_type}/
    
elif [ $data_type = train ]; then

    python ./esnli_bert2bert_eval.py\
    -model_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_partition${partition}_train${train_size}_${quality}/best_model/ \
    -eval_data_path ${data_path_prefix}seed${seed}/partition${partition}/${data_type}_${train_size}_${quality}.csv \
    -eval_results_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_partition${partition}_train${train_size}_${quality}/eval_${data_type}/

fi

 