python ../run_glue.py \
	--model_name_or_path ./save_best_model/esnli_phe_emptyexpl_ft_HANS30T300_outputs/ \
	--task_name ESNLI \
	--do_eval \
	--data_dir ./esnli_train_dev_empty_expl/ \
	--max_seq_length 128 \
	--output_dir ./tmp/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b \
	--output_error_file 
