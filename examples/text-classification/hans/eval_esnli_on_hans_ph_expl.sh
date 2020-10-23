python ../run_glue.py \
	--model_name_or_path ../esnli/esnli_phe_1empty_outputs_copy/ \
	--task_name HANS \
	--do_eval \
	--data_dir /data/rosa/data/hans/with_rosa_expl_1k_in_esnli_format/lex_prep_templates/high_quality_expl/ \
	--max_seq_length 128 \
	--output_dir ../esnli/esnli_phe_1empty_outputs_copy/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b 
