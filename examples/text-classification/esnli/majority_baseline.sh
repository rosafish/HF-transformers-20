seed=0

for partition in `seq 0 4`
	do
	for test_type in ivit ovit ivot ovot
		do 
			python majority_baseline.py /net/scratch/zhouy1/data/generated_data/seed${seed}/partition${partition}/ 1 $test_type
			python majority_baseline.py /net/scratch/zhouy1/data/generated_data/seed${seed}/partition${partition}/ 2 $test_type
			python majority_baseline.py /net/scratch/zhouy1/data/generated_data/seed${seed}/partition${partition}/ 4 $test_type
			python majority_baseline.py /net/scratch/zhouy1/data/generated_data/seed${seed}/partition${partition}/ 8 $test_type
			python majority_baseline.py /net/scratch/zhouy1/data/generated_data/seed${seed}/partition${partition}/ 16 $test_type
			python majority_baseline.py /net/scratch/zhouy1/data/generated_data/seed${seed}/partition${partition}/ 32 $test_type
			python majority_baseline.py /net/scratch/zhouy1/data/generated_data/seed${seed}/partition${partition}/ 64 $test_type
		done
	done

