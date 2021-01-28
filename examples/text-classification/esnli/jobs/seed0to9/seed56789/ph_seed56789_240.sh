cd ../..

train_size=240
dev_size=48

for seed in `seq 5 9`
do
	sh run_bert_ph.sh $seed $train_size $dev_size
	sh run_esnli_ph.sh $seed $train_size $dev_size

	for test_type in mvmt misvmt mvmist misvmist
	do
		sh eval_bert_ph.sh $seed $train_size $test_type
		sh eval_esnli_ph.sh $seed $train_size $test_type

	done
done

