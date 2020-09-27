python ../run_glue.py \
	--model_name_or_path ./esnli_expl1_outputs_copy/ \
	--task_name ESNLI \
	--do_eval \
	--data_dir TODO \
	--max_seq_length 128 \
	--output_dir ./esnli_expl1_outputs_copy/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type expl1:a 
