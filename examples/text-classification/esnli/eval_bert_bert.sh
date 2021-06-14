seqclas_pretrained_model=bert #bert or esnli
bert2bert_pretrained_model=bert #bert or esnli

# inputs
quality=$1
seed=$2
partition=$3
test_type=$4 # mvmt or misvmt or mvmist or misvmist
training_size=$5
data_dir_name=$6
validation=$7 # gold or generated
train=$8 # gold or generated

server=uchi # ego or uchi

if [ $server = ego ]; then

	data_path_prefix=/data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/
    bert2bert_gen_data_path_prefix=../../EncoderDecoderModel/esnli/save_best_models/
    model_path_prefix=./save_best_model/${data_dir_name}_etp_train_${train}_dev_${validation}/

elif [ $server = uchi ]; then

	data_path_prefix=/net/scratch/zhouy1/data/${data_dir_name}/
    bert2bert_gen_data_path_prefix=/net/scratch/zhouy1/randomness_experiment/${data_dir_name}/edm/
    model_path_prefix=/net/scratch/zhouy1/randomness_experiment/${data_dir_name}/seqclas_train_${train}_dev_${validation}/

	cp ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}_datafrom${bert2bert_pretrained_model}/vocab.txt ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}_datafrom${bert2bert_pretrained_model}/best_model/

fi

if [ $validation = gold ]; then

    dev_data_path=${data_path_prefix}seed${seed}/partition${partition}/test_${test_type}_300_${quality}.csv

elif [ $validation = generated ]; then

    dev_data_path=${bert2bert_gen_data_path_prefix}${bert2bert_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}/${test_type}_test_text_esnli_format.csv

fi

python ../run_glue.py \
	--model_name_or_path ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}_datafrom${bert2bert_pretrained_model}/best_model/ \
	--task_name HANS \
	--do_eval \
	--dev_data_path $dev_data_path \
	--max_seq_length 128 \
	--output_dir ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}_datafrom${bert2bert_pretrained_model}/eval_${test_type}_test/ \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b \
	--test_data_info ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}_datafrom${bert2bert_pretrained_model}/eval_${test_type}_test_by_example.csv
