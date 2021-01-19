cd ..

train_size=240
dev_size=48

for seed in `seq 0 4`
do

	for test_type in mvmt misvmt mvmist misvmist
	do
		sh run_bert_ph.sh $seed $train_size $dev_size
		sh eval_bert_ph.sh $seed $train_size $test_type
	
		sh run_esnli_ph.sh $seed $train_size $dev_size
		sh eval_esnli_ph.sh $seed $train_size $test_type

	done
done

