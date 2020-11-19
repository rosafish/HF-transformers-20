python ./esnli_bert2bert_train.py\
    -data_dir /data/rosa/data/hans/in_esnli_format/template_expls/30T/with_unseen/1500/high_q_expl/ \
    -cached_train_features_file ../cache/cached_train_hans30T_1500train_unseen_highqexpl \
    -save_trained_model_dir ./save_best_models/ft_bert_on_hans30T_1500train_unseen_highqexpl/ \
    -max_steps 4000 