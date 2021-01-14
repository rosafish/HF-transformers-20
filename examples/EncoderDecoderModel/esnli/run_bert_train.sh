pretrained_model=bert # esnli or bert
train_size=240
dev_size=48
quality=$1
seed=$2
server=uchi # ego or uchi
debug=false

if [ $pretrained_model = esnli ]; then

    model_dir=./save_best_models/esnli_train_trained_model/best_model/

elif [ $pretrained_model = bert ]; then

    model_dir=bert-base-uncased

fi

if [ $server = ego ]; then

    data_path_prefix=/data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/
    save_model_path_prefix=./save_best_models/

elif [ $server = uchi ]; then

    data_path_prefix=~/data/randomness_experiment/
    save_model_path_prefix=/net/scratch/zhouy1/randomness_experiment/edm/

fi

if [ $debug = true ]; then

    pretrained_model=debug

fi

if [ $train_size -lt 2000 ]; then

    max_steps=2000
    eval_steps=200
    eval_method=step

    python ./esnli_bert2bert_train.py\
    -model_dir $model_dir \
    -train_data_path ${data_path_prefix}seed${seed}/train${train_size}_${quality}.csv \
    -eval_data_path ${data_path_prefix}seed${seed}/dev${dev_size}_${quality}.csv \
    -cached_train_features_file ../cache/${pretrained_model}_hans_seed${seed}_train${train_size}_${quality} \
    -save_trained_model_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_train${train_size}_${quality}/ \
    -max_steps $max_steps \
    -eval_method $eval_method \
    -eval_steps $eval_steps 

else

    train_epochs=10
    eval_method=epoch

    python ./esnli_bert2bert_train.py\
    -model_dir $model_dir \
    -train_data_path ${data_path_prefix}seed${seed}/train${train_size}_${quality}.csv \
    -eval_data_path ${data_path_prefix}seed${seed}/dev${dev_size}_${quality}.csv \
    -cached_train_features_file ../cache/${pretrained_model}_hans_seed${seed}_train${train_size}_${quality} \
    -save_trained_model_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_train${train_size}_${quality}/ \
    -eval_method $eval_method \
    -train_epochs $train_epochs 

fi





