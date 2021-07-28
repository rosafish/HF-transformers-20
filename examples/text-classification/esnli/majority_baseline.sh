seed=0
data_dir_name=$1

for partition in `seq 0 4`
do
	for test_type in ivit ovit ivot ovot
	do 
	    for train_size in 1 2 4 8 16
		do
			python majority_baseline.py /net/scratch/zhouy1/data/${data_dir_name}/seed${seed}/partition${partition}/ $train_size $test_type
		done
	done
done

