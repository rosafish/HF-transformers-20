cd ..

for partition in `seq 1 4`
do
	sh run_bert_ph.sh 1 $partition 1 1 generated_data_new_setting
	sh run_bert_ph.sh 1 $partition 2 1 generated_data_new_setting
	sh run_bert_ph.sh 1 $partition 4 1 generated_data_new_setting
	sh run_bert_ph.sh 1 $partition 8 2 generated_data_new_setting
	sh run_bert_ph.sh 1 $partition 16 4 generated_data_new_setting
done
