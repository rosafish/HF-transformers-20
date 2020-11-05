python ./esnli_bert2bert_train.py\
    -data_dir /data/rosa/data/hans/in_esnli_format/template_expls/train_500/low_q_expl/ \
    -cached_train_features_file ../cache/cached_train_hans500_1temp_lowqexpl \
    -save_trained_model_dir ./save_best_models/ft_esnli_on_hans500_1temp_lowqexpl/ \
    -max_steps 2000 