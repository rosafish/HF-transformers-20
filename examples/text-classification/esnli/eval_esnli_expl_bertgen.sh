quality=ex_low
seed=0
training_size=240
test_type=matched
seqclas_pretrained_model=esnli #bert or esnli
bert2bert_pretrained_model=esnli #bert or esnli

python ../run_glue.py \
	--model_name_or_path ./save_best_model/${seqclas_pretrained_model}_hans_seed${seed}_train${training_size}_${quality}_datafrom${bert2bert_pretrained_model}/best_model/ \
	--task_name ESNLI \
	--do_eval \
	--dev_data_path /data/rosa/HF-transformers-20/examples/EncoderDecoderModel/esnli/save_best_models/${bert2bert_pretrained_model}_hans_seed${seed}_train${training_size}_${quality}/${test_type}_test_text_esnli_format.csv \
	--max_seq_length 128 \
	--output_dir ./save_best_model/${seqclas_pretrained_model}_hans_seed${seed}_train${training_size}_${quality}_datafrom${bert2bert_pretrained_model}/eval_${test_type}_test/ \
	--overwrite_cache \
	--esnli_input_type p+h:a,expl1:b 
