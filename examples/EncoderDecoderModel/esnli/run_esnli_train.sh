pretrained_model = esnli # esnli or bert

if [ $pretrained_model = esnli ]; then

    model_dir = ./save_best_models/esnli_train_trained_model/best_model/

elif [ $pretrained_model = bert ]; then

    model_dir = bert-base-uncased

fi

python ./esnli_bert2bert_train.py\
    -model_dir $model_dir \
    -train_data_path ~/data/randomness_experiment/seed1/train12000_high.csv \
    -eval_data_path ~/data/randomness_experiment/seed1/dev2400_high.csv \
    -cached_train_features_file ../cache/bert_hans_seed1_train12000_high \
    -save_trained_model_dir /net/scratch/zhouy1/randomness_experiment/edm/bert_hans_seed1_train12000_high/ \
    -max_steps 10000 \
    -eval_method step \
    -eval_steps 2000 
