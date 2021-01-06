python ./esnli_bert2bert_train.py\
    -model_dir bert-base-uncased \
    -train_data_path ~/data/randomness_experiment/seed1/train12000_ex_low.csv \
    -eval_data_path ~/data/randomness_experiment/seed1/dev2400_ex_low.csv \
    -cached_train_features_file ../cache/bert_hans_seed1_train12000_ex_low \
    -save_trained_model_dir /net/scratch/zhouy1/randomness_experiment/edm/bert_hans_seed1_train12000_ex_low/ \
    -max_steps 10000 \
    -eval_method step \
    -eval_steps 2000 
