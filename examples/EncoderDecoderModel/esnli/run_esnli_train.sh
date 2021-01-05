python ./esnli_bert2bert_train.py\
    -model_dir bert-base-uncased \
    -train_data_path /data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/seed0/train240_low.csv \
    -eval_data_path /data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/seed0/dev2400_low.csv \
    -cached_train_features_file ../cache/bert_hans_seed0_train240_low \
    -save_trained_model_dir ./save_best_models/bert_hans_seed0_train240_low/ \
    -max_steps 4000 \
    -eval_method step \
    -eval_steps 1000 