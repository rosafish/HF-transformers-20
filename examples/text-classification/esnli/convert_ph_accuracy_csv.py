import sys
import csv
import glob
sys.path.append('/data/rosa/my_github/misinformation/code/')

if __name__=="__main__":
    rows = []

    header = ['bert mvmt', 'bert misvmt', 'bert mvmist', 'bert misvmist', \
              'esnli mvmt', 'esnli misvmt', 'esnli mvmist', 'esnli misvmist', \
              ]

    train_size = 7680
    output_csv = './randomness_exp_ph_train%s.csv' % train_size

    for seed in range(5):
        row = []
        for model in ['bert', 'esnli']:
            for test_type in ['mvmt', 'misvmt', 'mvmist', 'misvmist']:
                path = './save_best_model/%s_hans_seed%d_train%d_empty_expl/eval_%s_test/eval_results_esnli.txt' % (model, seed, train_size, test_type)

                f = open(path, "r")
                line1 = f.readline()
                line2 = f.readline()

                line2_list = line2.split(" = ")
                accuracy = eval(line2_list[1])
                accuracy = round(accuracy*100, 3)

                row.append(accuracy)

        rows.append(row)

    from myTools import write_csv
    write_csv(output_csv, rows, header)