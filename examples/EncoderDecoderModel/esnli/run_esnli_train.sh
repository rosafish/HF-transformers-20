python ./esnli_bert2bert_train.py\
    -model_dir bert-base-uncased \
    -train_data_path /data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/seed0/train2400_high.csv \
    -eval_data_path /data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/seed0/dev2400_high.csv \
    -cached_train_features_file ../cache/bert_hans_seed0_train2400_high \
    -save_trained_model_dir ./save_best_models/bert_hans_seed0_train2400_high/ \
    -max_steps 10000 \
    -eval_method step \
    -eval_steps 2000 
