seqclas_pretrained_model=bert #bert or esnli

#inputs
seed=$1
training_size=$2
dev_size=$3

quality=empty_expl # fixed for p+h benchmark
test_type=mvmt # does not affect anything
test_size=3000

server=ego # ego or uchi

if [ $server = ego ]; then

    data_path_prefix=/data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/
    bert2bert_gen_data_path_prefix=../../EncoderDecoderModel/esnli/save_best_models/
    model_path_prefix=./save_best_model/

elif [ $server = uchi ]; then

    data_path_prefix=~/data/randomness_experiment/
    bert2bert_gen_data_path_prefix=/net/scratch/zhouy1/randomness_experiment/edm/
    model_path_prefix=/net/scratch/zhouy1/randomness_experiment/seqclas/

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
	--train_data_path ${data_path_prefix}seed${seed}/train${training_size}_${quality}.csv \
	--dev_data_path ${data_path_prefix}seed${seed}/dev${dev_size}_${quality}.csv \
	--test_data_path ${data_path_prefix}seed${seed}/${test_type}${test_size}_${quality}.csv \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--output_dir ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_train${training_size}_${quality}/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b \
	--save_best_model \
	--eval_method step \
	--max_steps 200 \
	--eval_steps 4
