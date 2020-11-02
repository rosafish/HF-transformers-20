python ./esnli_bert2bert_train.py\
    -data_dir /data/rosa/data/hans/in_esnli_format/template_expls/train_1k/low_q_expl/ \
    -cached_train_features_file ../cache/cached_train_hans1k_1temp_lowqexpl \
    -save_trained_model_dir ./save_best_models/ft_esnli_on_hans1k_1temp_lowqexpl/ \
    -max_steps 2000 