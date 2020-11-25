python ./esnli_bert2bert_eval.py\
    -model_dir ./save_best_models/esnli_train_trained_model/best_model/ \
    -eval_data_path /data/rosa/data/glue_mnli_esnli_format/matched_dev/esnli_train.csv \
    -generate_expl_on_training_data