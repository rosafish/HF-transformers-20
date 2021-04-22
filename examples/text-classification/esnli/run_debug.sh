bert2bert_pretrained_model=bert #bert or esnli
seqclas_pretrained_model=bert #bert or esnli

#inputs
quality=nl
seed=0
partition=0
training_size=1

server=ego # ego or uchi

if [ $server = ego ]; then

    data_path_prefix=/data/rosa/hans-forked/auto/generated_data/
    bert2bert_gen_data_path_prefix=../../EncoderDecoderModel/esnli/save_best_models/
    model_path_prefix=./save_best_model/

elif [ $server = uchi ]; then

    data_path_prefix=~/data/randomness_experiment/
    bert2bert_gen_data_path_prefix=/net/scratch/zhouy1/randomness_experiment/edm/
    model_path_prefix=/net/scratch/zhouy1/randomness_experiment/seqclas/

fi

if [ $seqclas_pretrained_model = esnli ]; then

    model_dir=./save_best_model/esnli_phe_goldexpl_outputs/best_model/

elif [ $seqclas_pretrained_model = bert ]; then

    model_dir=bert-base-uncased

fi

python ../run_glue.py \
	--model_name_or_path $model_dir \
	--task_name ESNLI \
	--do_train \
	--do_eval \
	--train_data_path ${data_path_prefix}seed${seed}/partition${partition}/train_${training_size}_${quality}.csv \
	--dev_data_path ${data_path_prefix}seed${seed}/partition${partition}/dev_1_${quality}.csv \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--output_dir ./debug/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b \
	--save_best_model \
	--eval_method step \
	--max_steps 200 \
	--eval_steps 4
