import csv
import sys

def get_template_ids_by_guid(input_csv):
    template_ids_by_guid = {}
    with open(input_csv, newline='') as f:
        reader = csv.reader(f)

        for (i, line) in enumerate(reader):
            if i == 0:
                continue

            guid = str(line[0])
            template_id = line[1]

            template_ids_by_guid[guid] = template_id
    return template_ids_by_guid


def get_acc(tf_list):
    success = 0
    for item in tf_list:
        if item == 'True':
            success += 1
    return round(success/len(tf_list), 5)*100


def get_acc_for_each_template_id(results_by_template_id):
    acc_by_template_id = {}
    for template_id in results_by_template_id.keys():
        acc_by_template_id[template_id] = get_acc(results_by_template_id[template_id])
    return acc_by_template_id


if __name__=='__main__': 
    # model input file directory, this file name should not contain explanation type
    # e.g.: /net/scratch/zhouy1/data/hard_test_templates/seed0/partition0/
    input_dir = sys.argv[1]
    
    # model generated explanations text file
    # e.g.: /net/scratch/zhouy1/randomness_experiment/hard_test_templates/label_only/bert_hans_seed0_partition0_train2_empty_expl
    output_dir = sys.argv[2] 

    test_type = sys.argv[3]

    input_csv = input_dir+'test_'+test_type+'_300.csv'
    by_exmaple_output_csv = output_dir+'eval_'+test_type+'_test_by_example.csv'
    by_template_output_path = output_dir+'eval_'+test_type+'_test_acc_by_temp.txt'

    template_ids_by_guid = get_template_ids_by_guid(input_csv)
    # <key, value>: <guid, template id>

    results_by_template_id = {}
    
    with open(by_exmaple_output_csv, newline='') as f:
        reader = csv.reader(f)

        for (i, line) in enumerate(reader):
            if i == 0:
                continue

            guid = str(line[0])
            result = str(line[1])
            template_id = str(template_ids_by_guid[guid])

            if template_id in results_by_template_id:
                results_by_template_id[template_id].append(result)
            else:
                results_by_template_id[template_id] = [result]
    
    acc_by_template_id = get_acc_for_each_template_id(results_by_template_id)

    f = open(by_template_output_path, 'w+')
    for template_id in acc_by_template_id.keys():
        print('template id: ', template_id)
        rounded_acc = acc_by_template_id[template_id]
        row = template_id + ', ' + str(rounded_acc) + '\n'
        f.write(row)
    f.close()
