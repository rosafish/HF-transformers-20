pretrained_model=bert #bert or esnli

# inputs
quality=$1
seed=$2
data_type=$3 # mvmt or mvmist or misvmt or misvmist or dev
train_size=$4
dev_size=$5

server=uchi # ego or uchi

if [ $server = ego ]; then

    data_path_prefix=/data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/
    model_path_prefix=./save_best_models/

elif [ $server = uchi ]; then

    data_path_prefix=~/data/randomness_experiment/
    model_path_prefix=/net/scratch/zhouy1/randomness_experiment/edm/

fi

dir=${model_path_prefix}${pretrained_model}_hans_seed${seed}_train${train_size}_${quality}/

if [ $data_type = dev ]; then
    # dev
    python convert_generated_embedding_text.py \
    -embedding_csv_path ${dir}eval_dev/epochNone*.csv \
    -text_csv_path ${dir}dev_text.csv 

    python convert_bertgen_to_original_format.py \
    -gold_expl_csv_path ${data_path_prefix}seed${seed}/dev${dev_size}_${quality}.csv \
    -output_csv_path ${dir}dev_text_esnli_format.csv \
    -bert_expl_csv_path ${dir}dev_text.csv 

elif [ $data_type = mvmt ] || [ $data_type = misvmt ]; then
    # matched templates
    python convert_generated_embedding_text.py \
    -embedding_csv_path ${dir}eval_${data_type}_test/epochNone*.csv \
    -text_csv_path ${dir}${data_type}_test_text.csv 

    python convert_bertgen_to_original_format.py \
    -gold_expl_csv_path ${data_path_prefix}seed${seed}/${data_type}_test12000_${quality}.csv \
    -output_csv_path ${dir}${data_type}_test_text_esnli_format.csv \
    -bert_expl_csv_path ${dir}${data_type}_test_text.csv 

elif [ $data_type = mvmist ] || [ $data_type = misvmist ]; then
    # mismatched templates
    python convert_generated_embedding_text.py \
    -embedding_csv_path ${dir}eval_${data_type}_test/epochNone*.csv \
    -text_csv_path ${dir}${data_type}_test_text.csv 

    python convert_bertgen_to_original_format.py \
    -gold_expl_csv_path ${data_path_prefix}seed${seed}/${data_type}_test3000_${quality}.csv \
    -output_csv_path ${dir}${data_type}_test_text_esnli_format.csv \
    -bert_expl_csv_path ${dir}${data_type}_test_text.csv 

fi