import sys
import csv

def sort_tuple(tup): 
    # reverse = None (Sorts in Ascending order) 
    # key is set to sort using second element of 
    # sublist lambda has been used 
    tup.sort(key = lambda x: x[1]) 
    return tup 

def find_worst_templates_id(bleu_by_temp_path, num_worst_temp):
    bleu_list = []
    with open(bleu_by_temp_path, newline='', 'r') as f:
        reader = csv.reader(f)
        for (i, line) in enumerate(reader):
            print(line)
            template_id = eval(line[0])
            bleu = round(eval(line[1]), 5)
            print("template_id: ", template_id)
            print("bleu: ", bleu)

            if i > 2:
                return

            bleu_list.append((template_id, bleu))

    bleu_list_sorted = sort_tuple(bleu_list)

    result = bleu_list_sorted[:5]
    print(result)
    return result

def main():
    seed = sys.argv[1]
    partition = sys.argv[2]
    expl_type = sys.argv[3]
    test_type = sys.argv[4]
    model = sys.argv[5]
    train_size = sys.argv[6]
    data_dir_name = 'before_new_setting'

    bleu_by_temp_path = '/net/scratch/zhouy1/randomness_experiment/%s/edm/%s_hans_seed%s_partition%s_train%s_%s/%s_test_bleu_by_temp.txt' % 
                        (data_dir_name, model, seed, partition, train_size, expl_type, test_type)

    # TODO: find worst five templates
    worst_templates_id = find_worst_templates_id(bleu_by_temp_path ,5)

    # TODO: replace Ns and Np to N
    # TODO: find its closest train template (not itself) by jaccard distance

if __name__=='__main__':
    main()