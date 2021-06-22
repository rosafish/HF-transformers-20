bert2bert_pretrained_model=esnli #bert or esnli
seqclas_pretrained_model=esnli #bert or esnli

#inputs
quality=$1
seed=$2
partition=$3
training_size=$4
dev_size=$5
data_dir_name=$6
validation=$7 # gold or generated
train=$8 # gold or generated

server=uchi # ego or uchi

if [ $server = ego ]; then

    data_path_prefix=/data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/
    bert2bert_gen_data_path_prefix=../../EncoderDecoderModel/esnli/save_best_models/
    model_path_prefix=./save_best_model/${data_dir_name}_etp_expl_only_train_${train}_dev_${validation}/

elif [ $server = uchi ]; then

    data_path_prefix=/net/scratch/zhouy1/data/${data_dir_name}/
    bert2bert_gen_data_path_prefix=/net/scratch/zhouy1/randomness_experiment/${data_dir_name}/edm/
    model_path_prefix=/net/scratch/zhouy1/randomness_experiment/${data_dir_name}/seqclas_expl_only_train_${train}_dev_${validation}/

fi

if [ $seqclas_pretrained_model = esnli ]; then

    model_dir=./save_best_model/esnli_phe_goldexpl_outputs/best_model/

elif [ $seqclas_pretrained_model = bert ]; then

    model_dir=bert-base-uncased

fi

if [ $validation = gold ]; then

    dev_data_path=${data_path_prefix}seed${seed}/partition${partition}/dev_${dev_size}_${quality}.csv

elif [ $validation = generated ]; then

    dev_data_path=${bert2bert_gen_data_path_prefix}${bert2bert_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}/dev_text_esnli_format.csv

fi

if [ $train = gold ]; then

    train_data_path=${data_path_prefix}seed${seed}/partition${partition}/train_${training_size}_${quality}.csv

elif [ $train = generated ]; then

    train_data_path=${bert2bert_gen_data_path_prefix}${bert2bert_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}/train_text_esnli_format.csv

fi

python ../run_glue.py \
	--model_name_or_path $model_dir \
	--task_name HANS \
	--do_train \
	--do_eval \
	--train_data_path $train_data_path \
	--dev_data_path $dev_data_path \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--output_dir ${model_path_prefix}${seqclas_pretrained_model}_hans_seed${seed}_partition${partition}_train${training_size}_${quality}_datafrom${bert2bert_pretrained_model}/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type expl1:a \
	--save_best_model \
	--eval_method step \
	--max_steps 200 \
	--eval_steps 4
