python ../run_glue.py \
	--model_name_or_path ./esnli_phe_1empty_outputs_copy/ \
	--task_name ESNLI \
	--do_eval \
	--data_dir ./esnli_train_on_gen_expl/eval_dev_emptyexpl/ \
	--max_seq_length 128 \
	--output_dir ./esnli_phe_1empty_outputs_copy/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b \
	--output_error_file 
