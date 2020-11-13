python ../run_glue.py \
	--model_name_or_path ./save_best_model/esnli_ph_expl_outputs/best_model/ \
	--task_name ESNLI \
	--do_train \
	--do_eval \
	--data_dir /data/rosa/data/hans/in_esnli_format/template_expls/30T_15000/high_q_expl/ \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 5.0 \
	--output_dir ./save_best_model/esnli_phe_gold_highqexpl_ft_HANS30T15000_outputs/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b \
	--save_best_model 
