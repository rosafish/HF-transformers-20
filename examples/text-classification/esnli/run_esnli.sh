quality=ex_low
seed=0
training_size=240

python ../run_glue.py \
	--model_name_or_path bert-base-uncased \
	--task_name ESNLI \
	--do_train \
	--do_eval \
	--train_data_path /data/rosa/data/hans/in_esnli_format/template_expls/randomness_experiment/seed${seed}/train${training_size}_${quality}.csv \
	--dev_data_path /data/rosa/HF-transformers-20/examples/EncoderDecoderModel/esnli/save_best_models/bert_hans_seed${seed}_train${training_size}_${quality}/dev_text_esnli_format.csv \
	--test_data_path /data/rosa/HF-transformers-20/examples/EncoderDecoderModel/esnli/save_best_models/bert_hans_seed${seed}_train${training_size}_${quality}/matched_test_text_esnli_format.csv \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--output_dir ./save_best_model/bert_hans_seed${seed}_train${training_size}_${quality}/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b \
	--save_best_model \
	--eval_method step \
	--max_steps 200 \
	--eval_steps 4
