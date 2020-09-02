python run_glue.py \
	--model_name_or_path bert-base-cased \
	--task_name ESNLI \
	--do_train \
	--do_eval \
	--data_dir /data/rosa/data/esnli \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 3.0 \
	--output_dir /data/rosa/HF-transformers-20/examples/text-classification/esnli_outputs/ \
	--overwrite_output_dir \
	--overwrite_cache 
