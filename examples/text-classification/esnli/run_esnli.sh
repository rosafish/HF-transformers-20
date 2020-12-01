python ../run_glue.py \
	--model_name_or_path ./save_best_model/esnli_phe_emptyexpl_outputs/ \
	--task_name ESNLI \
	--do_train \
	--do_eval \
	--data_dir /data/rosa/data/glue_mnli_esnli_format/matched_dev/ \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--output_dir ./save_best_model/esnli_mnli0shot_phe_emptyexpl_matched_dev_outputs/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b \
	--save_best_model \
	--eval_method step \
	--num_train_epochs 5.0 \
	--eval_steps 1000
