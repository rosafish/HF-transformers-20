python ./esnli_bert2bert_train.py\
    -model_dir ./save_best_models/esnli_train_trained_model/best_model/ \
    -data_dir /data/rosa/data/glue_mnli_esnli_format_30/matched_dev/ \
    -cached_train_features_file ../cache/mnli30_matched_dev \
    -save_trained_model_dir ./save_best_models/debug_mnli30_matched_dev \
    -train_epochs 5 \
    -eval_method epoch \
    -eval_esnli_dev 