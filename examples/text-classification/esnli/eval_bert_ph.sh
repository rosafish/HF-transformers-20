seqclas_pretrained_model=bert #bert or esnli

# inputs
seed=$1
training_size=$2
test_type=$3 # mvmt or misvmt or mvmist or misvmist

quality=empty_expl # fixed for p+h benchmark
server=ego # ego or uchi

if [ $server = ego ]; then

	data_path_prefix=/data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/
    bert2bert_gen_data_path_prefix=../../EncoderDecoderModel/esnli/save_best_models/
    model_path_prefix=./save_best_model/

	cp ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_train${training_size}_${quality}/vocab.txt ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_train${training_size}_${quality}/best_model/

elif [ $server = uchi ]; then

	data_path_prefix=~/data/randomness_experiment/
    bert2bert_gen_data_path_prefix=/net/scratch/zhouy1/randomness_experiment/edm/
    model_path_prefix=/net/scratch/zhouy1/randomness_experiment/seqclas/

fi

if [ $test_type = mvmt ] || [ $test_type = misvmt ]; then

    test_size=12000

elif [ $test_type = mvmist ] || [ $test_type = misvmist ]; then

    test_size=3000

fi

python ../run_glue.py \
	--model_name_or_path ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_train${training_size}_${quality}/best_model/ \
	--task_name ESNLI \
	--do_eval \
	--dev_data_path ${data_path_prefix}seed${seed}/${test_type}_test${test_size}_${quality}.csv \
	--max_seq_length 128 \
	--output_dir ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_train${training_size}_${quality}/eval_${test_type}_test/ \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b 
