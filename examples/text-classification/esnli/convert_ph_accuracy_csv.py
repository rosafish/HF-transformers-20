import sys
import csv
import glob
sys.path.append('/data/rosa/my_github/misinformation/code/')
from myTools import write_csv

if __name__=="__main__":
    train_size = int(sys.argv[1])
    data_dir_name = sys.argv[2]

    rows = []

    header = ['bert ivit', 'bert ovit', 'bert ivot', 'bert ovot', \
              'esnli ivit', 'esnli ovit', 'esnli ivot', 'esnli ovot', \
              ]

    output_csv = './randomness_exp_ph_train%s_%s.csv' % (train_size, data_dir_name)

    for seed in range(1,2):
        for partition in range(5):
            row = []
            for model in ['bert', 'esnli']:
                for test_type in ['ivit', 'ovit', 'ivot', 'ovot']:
                    path = './save_best_model/%s_hans_seed%d_partition%d_train%d_empty_expl/eval_%s_test/eval_results_hans.txt' % (model, seed, partition, train_size, test_type)

                    print(path)

                    try:
                        f = open(path, "r")
                        line1 = f.readline()
                        line2 = f.readline()

                        line2_list = line2.split(" = ")
                        accuracy = eval(line2_list[1])
                        accuracy = round(accuracy*100, 3)
                    except:
                        accuracy=-1
                    row.append(accuracy)

            rows.append(row)

    write_csv(output_csv, rows, header)