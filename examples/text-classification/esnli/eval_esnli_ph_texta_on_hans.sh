python run_glue.py \
	--model_name_or_path ./esnli/esnli_ph_texta_outputs_copy/ \
	--task_name HANS \
	--do_eval \
	--data_dir /data/rosa/data/hans/in_esnli_format/ \
	--max_seq_length 128 \
	--output_dir ./esnli/esnli_ph_texta_outputs_copy/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type doesnotmatter 
