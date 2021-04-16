import csv
import sys
sys.path.append('/home/zhouy1/misinformation/code/')
from myTools import write_csv

if __name__=='__main__':
    data_dir = sys.argv[1]
    train_size = sys.argv[2]
    test_type = sys.argv[3]

    train_data_path = '%strain_%s.csv' % (data_dir, train_size)
    eval_data_path = '%stest_%s_300.csv' % (data_dir, test_type)
    output_csv_path = '%seval_test_%s.csv' % (data_dir, test_type)
    output_acc_path = '%seval_test_%s_acc.txt' % (data_dir, test_type)

    labels_train = {'entailment': 0, 'neutral': 0}

    # find majority class in train_data
    with open(train_data_path, 'r') as f:
        f_csv = csv.reader(f)
        for (i, line) in enumerate(f_csv):
            #print(i)
            #print(line)
            label=line[5]
            #print(label)
            if label in labels_train:
                labels_train[label] += 1

    print(labels_train)
    majority = 'entailment' if labels_train['entailment'] >= labels_train['neutral'] else 'neutral'

    print('majority: ', majority)
    output_headers = ['eval_guid','correctness']
    rows = []

    # output file of 0(miss) and 1(hit) when eval on eval_data    
    with open(eval_data_path, 'r') as f:
        f_csv = csv.reader(f)
        correct_count = 0
        total_count = 0
        for (i, line) in enumerate(f_csv):
            #print(i)
            #print(line)
            eval_guid=line[0]
            label=line[5]
            correctness = 1 if label == majority else 0
            row = [eval_guid, correctness]
            rows.append(row)

            correct_count += correctness
            total_count += 1

    write_csv(output_csv_path, rows, output_headers)
    print(correct_count)
    print(total_count)
    acc =  correct_count/total_count
    print('acc: ', acc)
    with open(output_acc_path, 'w+') as f:
        f.write('acc: %f' % acc)

