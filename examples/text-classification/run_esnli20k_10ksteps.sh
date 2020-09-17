python run_glue.py \
	--model_name_or_path bert-base-cased \
	--task_name ESNLI \
	--do_train \
	--do_eval \
	--data_dir /data/rosa/data/esnli_20k/ \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--output_dir /data/rosa/HF-transformers-20/examples/text-classification/esnli20k_ph_texta_outputs_10ksteps/ \
	--overwrite_output_dir \
	--overwrite_cache \
	--esnli_input_type p+h:a \
	--max_steps 10000
