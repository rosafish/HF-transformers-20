# inputs
data_type=dev
input_file_name=epoch266*.csv
quality=ex_low

dir=./save_best_models/bert_hans_seed0_train240_${quality}/

if [ $data_type = dev ]; then
    # dev
    python convert_generated_embedding_text.py \
    -embedding_csv_path ${dir}${input_file_name} \
    -text_csv_path ${dir}dev_text.csv 

    python convert_bertgen_to_original_format.py \
    -gold_expl_csv_path /data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/seed0/dev2400_${quality}.csv \
    -output_csv_path ${dir}dev_text_esnli_format.csv \
    -bert_expl_csv_path ${dir}dev_text.csv 

elif [ $data_type = matched_test ]; then
    # matched_test
    python convert_generated_embedding_text.py \
    -embedding_csv_path ${dir}eval_matched_test/epochNone*.csv \
    -text_csv_path ${dir}matched_test_text.csv 

    python convert_bertgen_to_original_format.py \
    -gold_expl_csv_path /data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/seed0/matched_test3000_${quality}.csv \
    -output_csv_path ${dir}matched_test_text_esnli_format.csv \
    -bert_expl_csv_path ${dir}matched_test_text.csv 

elif [ $data_type = mismatched_test ]; then
    # mismatched_test
    python convert_generated_embedding_text.py \
    -embedding_csv_path ${dir}eval_mismatched_test/epochNone*.csv \
    -text_csv_path ${dir}mismatched_test_text.csv 

    python convert_bertgen_to_original_format.py \
    -gold_expl_csv_path /data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/seed0/mismatched_test3000_${quality}.csv \
    -output_csv_path ${dir}mismatched_test_text_esnli_format.csv \
    -bert_expl_csv_path ${dir}mismatched_test_text.csv 

fi