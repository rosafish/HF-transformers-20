python ./esnli_bert2bert_train.py\
    -model_dir bert-base-uncased \
    -train_data_path ~/data/randomness_experiment/seed0/train1200_low.csv \
    -eval_data_path ~/data/randomness_experiment/seed0/dev2400_low.csv \
    -cached_train_features_file ../cache/bert_hans_seed0_train1200_low \
    -save_trained_model_dir ./save_best_models/bert_hans_seed0_train1200_low/ \
    -max_steps 4000 \
    -eval_method step \
    -eval_steps 1000 
