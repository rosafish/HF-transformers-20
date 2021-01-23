seqclas_pretrained_model=bert #bert or esnli

# inputs
seed=$1
test_type=$2 # mvmt or misvmt or mvmist or misvmist

quality=empty_expl # fixed for p+h benchmark
server=ego # ego or uchi

if [ $server = ego ]; then

	data_path_prefix=/data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/

elif [ $server = uchi ]; then

	data_path_prefix=~/data/randomness_experiment/

fi

if [ $test_type = mvmt ] || [ $test_type = misvmt ]; then

    test_size=12000

elif [ $test_type = mvmist ] || [ $test_type = misvmist ]; then

    test_size=3000

fi

if [ $seqclas_pretrained_model = esnli ]; then

    model_dir=./save_best_model/esnli_phe_goldexpl_outputs/best_model/

elif [ $seqclas_pretrained_model = bert ]; then

    model_dir=bert-base-uncased

fi

python ../../run_glue.py \
	--model_name_or_path $model_dir \
	--task_name ESNLI \
	--do_eval \
	--dev_data_path ${data_path_prefix}seed${seed}/${test_type}_test${test_size}_${quality}.csv \
	--max_seq_length 128 \
	--output_dir ./${seqclas_pretrained_model}_ph_hans_seed${seed}/eval_${test_type}_test/ \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b 
