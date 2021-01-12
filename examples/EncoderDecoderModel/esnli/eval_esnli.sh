quality=high
test_type=mismatched
seed=0
train_size=240

python ./esnli_bert2bert_eval.py\
    -model_dir ./save_best_models/bert_hans_seed${seed}_train${train_size}_${quality}/best_model/ \
    -eval_data_path /data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/seed${seed}/${test_type}_test3000_${quality}.csv \
    -eval_results_dir ./save_best_models/bert_hans_seed${seed}_train${train_size}_${quality}/eval_${test_type}_test/ 