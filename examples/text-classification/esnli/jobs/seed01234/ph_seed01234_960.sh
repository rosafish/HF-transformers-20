cd ..

train_size=960
dev_size=192

for seed in `seq 0 4`
do
	sh run_bert_ph.sh $seed $train_size $dev_size
	sh run_esnli_ph.sh $seed $train_size $dev_size

	for test_type in mvmt misvmt mvmist misvmist
	do
		sh eval_bert_ph.sh $seed $train_size $test_type
		sh eval_esnli_ph.sh $seed $train_size $test_type

	done
done

