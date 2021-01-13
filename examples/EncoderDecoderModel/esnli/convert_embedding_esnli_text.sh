# inputs
data_type=dev
input_file_name=epoch266*.csv
quality=low
pretrained_model=esnli #bert or esnli
server=uchi # ego or uchi
seed=0
train_size=240

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
    -gold_expl_csv_path ${data_path_prefix}seed${seed}/dev2400_${quality}.csv \
    -output_csv_path ${dir}dev_text_esnli_format.csv \
    -bert_expl_csv_path ${dir}dev_text.csv 

elif [ $data_type = matched_test ]; then
    # matched_test
    python convert_generated_embedding_text.py \
    -embedding_csv_path ${dir}eval_matched_test/epochNone*.csv \
    -text_csv_path ${dir}matched_test_text.csv 

    python convert_bertgen_to_original_format.py \
    -gold_expl_csv_path ${data_path_prefix}seed${seed}/matched_test3000_${quality}.csv \
    -output_csv_path ${dir}matched_test_text_esnli_format.csv \
    -bert_expl_csv_path ${dir}matched_test_text.csv 

elif [ $data_type = mismatched_test ]; then
    # mismatched_test
    python convert_generated_embedding_text.py \
    -embedding_csv_path ${dir}eval_mismatched_test/epochNone*.csv \
    -text_csv_path ${dir}mismatched_test_text.csv 

    python convert_bertgen_to_original_format.py \
    -gold_expl_csv_path ${data_path_prefix}seed${seed}/mismatched_test3000_${quality}.csv \
    -output_csv_path ${dir}mismatched_test_text_esnli_format.csv \
    -bert_expl_csv_path ${dir}mismatched_test_text.csv 

fi