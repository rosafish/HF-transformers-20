seqclas_pretrained_model=bert #bert or esnli

# inputs
seed=$1
partition=$2
training_size=$3
test_type=$4 # ivit, ivot, ovit, ovot
data_dir_name=$5

quality=empty_expl # fixed for p+h benchmark
server=uchi # ego or uchi

test_size=300

if [ $server = ego ]; then

	data_path_prefix=/data/rosa/hans-forked/auto/${data_dir_name}/
    model_path_prefix=./save_best_model/

elif [ $server = uchi ]; then

    data_path_prefix=/net/scratch/zhouy1/data/${data_dir_name}/
    model_path_prefix=/net/scratch/zhouy1/randomness_experiment/${data_dir_name}/label_only/

fi

cp ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}/vocab.txt ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}/best_model/

python ../run_glue.py \
	--model_name_or_path ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}/best_model/ \
	--task_name HANS \
	--do_eval \
	--dev_data_path ${data_path_prefix}seed${seed}/partition${partition}/test_${test_type}_${test_size}_${quality}.csv \
	--max_seq_length 128 \
	--output_dir ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}/eval_${test_type}_test/ \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b \
	--test_data_info ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}/eval_${test_type}_test_by_templates.csv
