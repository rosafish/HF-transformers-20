python ./esnli_bert2bert_train.py\
    -data_dir /data/rosa/data/hans/in_esnli_format/template_expls/train_1k/high_q_expl/ \
    -cached_train_features_file ../cache/cached_train_hans1k_1temp_highqexpl \
    -save_trained_model_dir ./save_best_models/ft_esnli_on_hans1k_1temp_highqexpl/ \
    -max_steps 2000 \
    -hans 