python ../run_glue.py \
	--model_name_or_path ./save_best_model/bert_hans_seed0_train240_low/best_model/ \
	--task_name ESNLI \
	--do_eval \
	--dev_data_path /data/rosa/HF-transformers-20/examples/EncoderDecoderModel/esnli/save_best_models/bert_hans_seed0_train240_low/matched_test_text_esnli_format.csv \
	--max_seq_length 128 \
	--output_dir ./save_best_model/bert_hans_seed0_train240_low/ \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b 
