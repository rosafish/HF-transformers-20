seqclas_pretrained_model=bert #bert or esnli

# inputs
seed=$1
partition=$2
training_size=$3
test_type=$4 # ivit, ivot, ovit, ovot

quality=empty_expl # fixed for p+h benchmark
server=uchi # ego or uchi

test_size=300

if [ $server = ego ]; then

	data_path_prefix=/data/rosa/hans-forked/auto/generated_data/
    model_path_prefix=./save_best_model/

elif [ $server = uchi ]; then

    data_path_prefix=/net/scratch/zhouy1/data/generated_data/
    model_path_prefix=/net/scratch/zhouy1/randomness_experiment/label_only/

fi

cp ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}/vocab.txt ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}/best_model/

python ../run_glue.py \
	--model_name_or_path ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}/best_model/ \
	--task_name ESNLI \
	--do_eval \
	--dev_data_path ${data_path_prefix}seed${seed}/partition${partition}/test_${test_type}_${test_size}_${quality}.csv \
	--max_seq_length 128 \
	--output_dir ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}/eval_${test_type}_test/ \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b 
