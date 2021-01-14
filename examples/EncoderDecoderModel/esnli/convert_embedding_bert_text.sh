pretrained_model=bert #bert or esnli

# inputs
quality=$1
seed=$2
data_type=$3 # matched, mismatched, or dev
train_size=$4
dev_size=$5
input_file_name=$6 # dev embedding file name

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
    -embedding_csv_path ${dir}${input_file_name} \
    -text_csv_path ${dir}dev_text.csv 

    python convert_bertgen_to_original_format.py \
    -gold_expl_csv_path ${data_path_prefix}seed${seed}/dev${dev_size}_${quality}.csv \
    -output_csv_path ${dir}dev_text_esnli_format.csv \
    -bert_expl_csv_path ${dir}dev_text.csv 

elif [ $data_type = matched ]; then
    # matched_test
    python convert_generated_embedding_text.py \
    -embedding_csv_path ${dir}eval_matched_test/epochNone*.csv \
    -text_csv_path ${dir}matched_test_text.csv 

    python convert_bertgen_to_original_format.py \
    -gold_expl_csv_path ${data_path_prefix}seed${seed}/matched_test12000_${quality}.csv \
    -output_csv_path ${dir}matched_test_text_esnli_format.csv \
    -bert_expl_csv_path ${dir}matched_test_text.csv 

elif [ $data_type = mismatched ]; then
    # mismatched_test
    python convert_generated_embedding_text.py \
    -embedding_csv_path ${dir}eval_mismatched_test/epochNone*.csv \
    -text_csv_path ${dir}mismatched_test_text.csv 

    python convert_bertgen_to_original_format.py \
    -gold_expl_csv_path ${data_path_prefix}seed${seed}/mismatched_test3000_${quality}.csv \
    -output_csv_path ${dir}mismatched_test_text_esnli_format.csv \
    -bert_expl_csv_path ${dir}mismatched_test_text.csv 

fi