python ../run_glue.py \
	--model_name_or_path ../esnli/esnli_ph_expl_outputs_copy/ \
	--task_name HANS \
	--do_eval \
	--data_dir ./EDM_esnli_hans_expl/ \
	--max_seq_length 128 \
	--output_dir ../esnli/esnli_ph_expl_outputs_copy/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b 
