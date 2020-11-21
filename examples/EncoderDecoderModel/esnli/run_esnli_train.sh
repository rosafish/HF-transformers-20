python ./esnli_bert2bert_train.py\
    -data_dir /data/rosa/data/hans/in_esnli_format/template_expls/30T/with_unseen/15000/low_q_expl/ \
    -cached_train_features_file ../cache/cached_train_hans30T_15000train_unseen_lowqexpl \
    -save_trained_model_dir ./save_best_models/ft_bert_on_hans30T_15000train_unseen_lowqexpl/ \
    -max_steps 10000 \
    -eval_method step \
    -eval_steps 2000 