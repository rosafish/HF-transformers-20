python ./esnli_bert2bert_train.py\
    -model_dir ./save_best_models/esnli_train_trained_model/best_model/ \
    -data_dir /data/rosa/data/hans/in_esnli_format/template_expls/30T/with_unseen/24_6_split/240train/low_q_expl/ \
    -cached_train_features_file ../cache/debug_240_highqexpl \
    -save_trained_model_dir ./save_best_models/debug_240_highqexpl \
    -max_steps 2000 \
    -eval_method step \
    -eval_steps 1000 