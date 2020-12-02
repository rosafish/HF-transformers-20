python ../run_glue.py \
	--model_name_or_path ./save_best_model/esnli_phe_goldexpl_outputs/best_model/ \
	--task_name ESNLI \
	--do_eval \
	--data_dir ./data/eval_mnli_ETP_matched_dev/ \
	--max_seq_length 128 \
	--output_dir ./tmp/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b \
	--output_error_file 
