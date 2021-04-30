seqclas_pretrained_model=bert #bert or esnli
bert2bert_pretrained_model=bert #bert or esnli

# inputs
quality=$1
seed=$2
partition=$3
test_type=$4 # mvmt or misvmt or mvmist or misvmist
training_size=$5
data_dir_name=$6

server=uchi # ego or uchi

if [ $server = ego ]; then

    bert2bert_gen_data_path_prefix=../../EncoderDecoderModel/esnli/save_best_models/
    model_path_prefix=./save_best_model/

elif [ $server = uchi ]; then

    bert2bert_gen_data_path_prefix=/net/scratch/zhouy1/randomness_experiment/${data_dir_name}/edm/
    model_path_prefix=/net/scratch/zhouy1/randomness_experiment/${data_dir_name}/seqclas/

	cp ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}_datafrom${bert2bert_pretrained_model}/vocab.txt ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}_datafrom${bert2bert_pretrained_model}/best_model/

fi

python ../run_glue.py \
	--model_name_or_path ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}_datafrom${bert2bert_pretrained_model}/best_model/ \
	--task_name ESNLI \
	--do_eval \
	--dev_data_path ${bert2bert_gen_data_path_prefix}${bert2bert_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}/${test_type}_test_text_esnli_format.csv \
	--max_seq_length 128 \
	--output_dir ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}_datafrom${bert2bert_pretrained_model}/eval_${test_type}_test/ \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b \
	--test_data_info /net/scratch/zhouy1/randomness_experiment/${data_dir_name}/seqclas/${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}_datafrom${bert2bert_pretrained_model}/eval_${test_type}_test_by_templates.csv
