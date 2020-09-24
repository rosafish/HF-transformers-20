python run_glue.py \
	--model_name_or_path ./esnli_expl1_outputs_copy/ \
	--task_name ESNLI \
	--do_eval \
	--data_dir /data/rosa/HF-transformers-20/examples/text-classification/hans/EDM_esnli_hans_expl/ \
	--max_seq_length 128 \
	--output_dir ./esnli_expl1_outputs_copy/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type expl1:a 
