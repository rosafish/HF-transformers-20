python ../run_glue.py \
	--model_name_or_path ./esnli25k_phe_1empty_2gold_outputs_copy/ \
	--task_name ESNLI \
	--do_eval \
	--data_dir /data/rosa/HF-transformers-20/examples/text-classification/esnli125k/EDM_esnli125k_expl/ \
	--max_seq_length 128 \
	--output_dir ./esnli25k_phe_1empty_2gold_outputs_copy/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b 
