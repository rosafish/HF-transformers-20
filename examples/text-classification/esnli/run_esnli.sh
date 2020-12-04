python ../run_glue.py \
	--model_name_or_path bert-base-uncased \
	--task_name ESNLI \
	--do_train \
	--do_eval \
	--data_dir ./data/mnli_bertgen_30_matched_dev/ \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--output_dir ./save_best_model/bert_mnli30shot_phe_goldpretrainedftbertexpl_matched_dev_outputs/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b \
	--save_best_model \
	--eval_method step \
	--max_steps 1000 \
	--eval_steps 20
