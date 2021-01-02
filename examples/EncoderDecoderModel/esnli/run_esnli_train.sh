python ./esnli_bert2bert_train.py\
    -model_dir bert-base-uncased \
    -train_data_path /data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/seed0/train240_ex_low.csv \
    -eval_data_path /data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/seed0/dev2400_ex_low.csv \
    -cached_train_features_file ../cache/bert_hans_seed0_train240_ex_low \
    -save_trained_model_dir ./save_best_models/bert_hans_seed0_train240_ex_low \
    -train_epochs 10 \
    -eval_method epoch 