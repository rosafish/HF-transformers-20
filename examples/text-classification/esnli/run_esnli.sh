python ../run_glue.py \
	--model_name_or_path ./save_best_model/esnli_phe_emptyexpl_outputs/best_model/ \
	--task_name ESNLI \
	--do_train \
	--do_eval \
	--data_dir /data/rosa/data/hans/in_esnli_format/template_expls/30T/with_unseen/1500/empty_expl/ \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 20.0 \
	--output_dir ./save_best_model/esnli_phe_emptyexpl_ft_HANS30T1500_unseen_outputs/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b \
	--save_best_model 
