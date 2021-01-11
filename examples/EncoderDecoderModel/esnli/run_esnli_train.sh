pretrained_model=esnli # esnli or bert
seed=0
train_size=240
quality=high
server=ego # ego or uchi

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

if [ $train_size < 2000 ]; then

    max_steps=4000
    eval_steps=1000

elif [ $train_size > 2000 ]; then

    max_steps=10000
    eval_steps=2000

elif [ $train_size = 2000 ]; then

    max_steps=10000
    eval_steps=2000

fi

python ./esnli_bert2bert_train.py\
    -model_dir $model_dir \
    -train_data_path ${data_path_prefix}seed${seed}/train${train_size}_${quality}.csv \
    -eval_data_path ${data_path_prefix}seed${seed}/dev2400_${quality}.csv \
    -cached_train_features_file ../cache/${pretrained_model}_hans_seed${seed}_train${train_size}_${quality} \
    -save_trained_model_dir ${save_model_path_prefix}${pretrained_model}_hans_seed${seed}_train${train_size}_${quality}/ \
    -max_steps $max_steps \
    -eval_method step \
    -eval_steps $eval_steps 
