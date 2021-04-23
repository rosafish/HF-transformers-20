seqclas_pretrained_model=bert #bert or esnli

#inputs
seed=$1
partition=$2
training_size=$3
dev_size=$4 
data_dir_name=$5

quality=empty_expl # fixed for p+h benchmark
test_type=mvmt # does not affect anything
test_size=300

server=uchi # ego or uchi

if [ $server = ego ]; then

    data_path_prefix=/data/rosa/hans-forked/auto/${data_dir_name}/
    model_path_prefix=./save_best_model/

elif [ $server = uchi ]; then

    data_path_prefix=/net/scratch/zhouy1/data/${data_dir_name}/
    model_path_prefix=/net/scratch/zhouy1/randomness_experiment/${data_dir_name}/label_only/

fi

if [ $seqclas_pretrained_model = esnli ]; then

    model_dir=./save_best_model/esnli_phe_emptyexpl_outputs/best_model/

elif [ $seqclas_pretrained_model = bert ]; then

    model_dir=bert-base-uncased

fi

python ../run_glue.py \
	--model_name_or_path $model_dir \
	--task_name ESNLI \
	--do_train \
	--do_eval \
	--train_data_path ${data_path_prefix}seed${seed}/partition${partition}/train_${training_size}_${quality}.csv \
	--dev_data_path ${data_path_prefix}seed${seed}/partition${partition}/dev_${dev_size}_${quality}.csv \
	--test_data_path ${data_path_prefix}seed${seed}/partition${partition}/test_${test_type}_${test_size}_${quality}.csv \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--output_dir ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b \
	--save_best_model \
	--eval_method step \
	--max_steps 1000 \
	--eval_steps 50
