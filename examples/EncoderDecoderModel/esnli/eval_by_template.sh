seed=0
data_dir_name=$1

for partition in `seq 0 4`
do
	for test_type in ivit ovit ivot ovot
	do
		for expl_type in nl pt
		do
			for model in bert esnli
			do
				for train_size in 1 2 4 8 16 32 64
				do
					python eval_by_template.py /net/scratch/zhouy1/data/${data_dir_name}/seed${seed}/partition${partition}/test_${test_type}_300.csv /net/scratch/zhouy1/randomness_experiment/edm/${model}_hans_seed${seed}_partition${partition}_train${train_size}_${expl_type}/ $expl_type $test_type
		
				done
			done
		done
	done
done
