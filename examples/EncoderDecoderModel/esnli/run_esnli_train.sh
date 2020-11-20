python ./esnli_bert2bert_train.py\
    -data_dir /data/rosa/data/hans/in_esnli_format/template_expls/30T/with_unseen/24_6_split/240train/high_q_expl/ \
    -cached_train_features_file ../cache/cached_train_hans30T_240train_unseen_highqexpl \
    -save_trained_model_dir ./save_best_models/ft_bert_on_hans30T_240train_unseen_highqexpl/ \
    -max_steps 2000 \
    -eval_method step \
    -eval_steps 500 