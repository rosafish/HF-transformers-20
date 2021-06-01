import sys
import csv
import editdistance

from nltk.tokenize import TweetTokenizer

var_type_subtypes={
    "N": ["Np", "Ns", "Nlocation"],
    "V": ["Vt", "Vi", "Vunderstand", "Vpp", "Vnpz", "Vnps", "Vconstquotentailed", "Vnonentquote"],
    "Adj": [],
    "Adv": ['Advoutent', 'Advent', 'Advembent', 'Advoutnent', 'Advnonent', 'Advembnent'],
    "Be": ['BePast'],
    "P": [],
    "Rels": [],
    "O": [],
    "Conj": [],
}

def sort_tuple(tup): 
    # reverse = None (Sorts in Ascending order) 
    # key is set to sort using second element of 
    # sublist lambda has been used 
    tup.sort(key = lambda x: x[1]) 
    return tup 

def find_worst_templates_id_by_bleu(bleu_by_temp_path, num_worst_temp):
    bleu_list = []
    all_test_templates_id = set()
    with open(bleu_by_temp_path, newline='') as f:
        reader = csv.reader(f)
        for (i, line) in enumerate(reader):
            # print(line)
            template_id = eval(line[0])
            bleu = round(eval(line[1]), 5)

            bleu_list.append((template_id, bleu))
            all_test_templates_id.add(template_id)

    bleu_list_sorted = sort_tuple(bleu_list)

    worst_temp_info = bleu_list_sorted[:num_worst_temp]
    print(worst_temp_info)
    return [t[0] for t in worst_temp_info], all_test_templates_id


def find_worst_templates_id_by_acc(pred_by_temp_path, num_worst_temp):
    acc_list = []
    all_test_templates_id = set()
    acc_sum = 0
    count = 0
    with open(pred_by_temp_path, newline='') as f:
        reader = csv.reader(f)
        for (i, line) in enumerate(reader):
            if i == 0:
                continue

            template_id = eval(line[0])
            acc = round(eval(line[1]), 5)

            acc_list.append((template_id, acc))
            all_test_templates_id.add(template_id)
            acc_sum += acc
            count += 1

    print('avg acc: ', acc_sum/count)

    acc_list_sorted = sort_tuple(acc_list)

    worst_temp_info = acc_list_sorted[:num_worst_temp]
    print(worst_temp_info)
    
    return [t[0] for t in worst_temp_info], all_test_templates_id
            

def replace_word_subtype2type(s):
    for k,v in var_type_subtypes.items():
        for subtype in v:
            s = s.replace(subtype, k)
    return s


def load_templates(templates_path, input_type):
    templates = []
    with open(templates_path, newline='') as f:
        reader = csv.reader(f)
        for (i, line) in enumerate(reader):
            if i == 0:
                continue

            p = replace_word_subtype2type(line[5])
            h = replace_word_subtype2type(line[6])
            
            if input_type == 'p':
                templates.append(p)
            elif input_type == 'h':
                templates.append(h)
            elif input_type == 'p+h':
                templates.append(p+h)
            else:
                print('invalid input type: ', input_type)
                return

    return templates

def get_jaccard_dist(s1, s2):
    tknzr = TweetTokenizer()

    s1 = set(tknzr.tokenize(s1))
    s2 = set(tknzr.tokenize(s2))
    
    # Find the intersection of words list of doc1 & doc2
    intersection = s1.intersection(s2)

    # Find the union of words list of doc1 & doc2
    union = s1.union(s2)
        
    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set
    return 1-float(len(intersection)) / len(union)

def worst_test_templates_by_bleu(data_dir_name, model, seed, partition, train_size, expl_type, test_type, num_worst_temp, input_type):
    bleu_by_temp_path = '/net/scratch/zhouy1/randomness_experiment/%s/edm/%s_hans_seed%s_partition%s_train%s_%s/%s_test_bleu_by_temp.txt' % \
                        (data_dir_name, model, seed, partition, train_size, expl_type, test_type)

    # find worst `num_worst_temp` templates
    worst_templates_id, all_test_templates_id = find_worst_templates_id_by_bleu(bleu_by_temp_path ,num_worst_temp)
    print('worst ', num_worst_temp, ' templates (id):', worst_templates_id)

    return worst_templates_id, all_test_templates_id


def worst_test_templates_by_acc(data_dir_name, model, seed, partition, train_size, expl_type, test_type, num_worst_temp, input_type):
    if expl_type == 'empty_expl':
        pred_by_temp_path = '/net/scratch/zhouy1/randomness_experiment/%s/label_only/%s_hans_seed%s_partition%s_train%s_empty_expl/eval_%s_test_acc_by_temp.txt' % \
                        (data_dir_name, model, seed, partition, train_size, test_type)
    else:
        pred_by_temp_path = '/net/scratch/zhouy1/randomness_experiment/%s/seqclas/%s_hans_seed%s_partition%s_train%s_%s_datafrom%s/eval_%s_test_acc_by_temp.txt' % \
                        (data_dir_name, model, seed, partition, train_size, expl_type, model, test_type)

    # find worst `num_worst_temp` templates
    worst_templates_id, all_test_templates_id = find_worst_templates_id_by_acc(pred_by_temp_path ,num_worst_temp)
    print('worst ', num_worst_temp, ' templates (id):', worst_templates_id)

    return worst_templates_id, all_test_templates_id

def main():
    seed = sys.argv[1]
    partition = sys.argv[2]
    expl_type = sys.argv[3]
    test_type = sys.argv[4]
    model = sys.argv[5]
    train_size = sys.argv[6]
    input_type = sys.argv[7] # p, h, or p+h
    num_worst_temp = int(sys.argv[8])
    standard = sys.argv[9] # standard: bleu or acc
    dist_measure = sys.argv[10]
    data_dir_name = sys.argv[11] # e.g. 'generated_data' or 'generated_data_new_setting'

    if standard == 'bleu':
        worst_templates_id, all_test_templates_id = worst_test_templates_by_bleu(data_dir_name, model, seed, partition, train_size, expl_type, test_type, num_worst_temp, input_type)
    elif standard == 'acc':
        worst_templates_id, all_test_templates_id = worst_test_templates_by_acc(data_dir_name, model, seed, partition, train_size, expl_type, test_type, num_worst_temp, input_type)

    all_templates_id = set([i for i in range(118)])
    if test_type == 'ivit' or test_type == 'ovit':
        all_train_templates_id = all_templates_id
    else:
        all_train_templates_id = all_templates_id - all_test_templates_id

    templates_path = "/net/scratch/zhouy1/hans-forked/auto/templates_new.csv"
    templates = load_templates(templates_path, input_type)

    for test_id in worst_templates_id:
        dist_list = []
        for train_id in all_train_templates_id:
            if train_id != test_id:
                s1 = templates[test_id].lower()
                s2 = templates[train_id].lower()
                tmp_dist = -1
                if dist_measure == 'jaccard':
                    tmp_dist = get_jaccard_dist(s1, s2)
                elif dist_measure == 'editdistance':
                    tmp_dist = editdistance.eval(s1, s2)
                else:
                    print('invalid distance measure')
                dist_list.append((train_id, tmp_dist))

        dist_list_sorted = sort_tuple(dist_list)
        closest_template_info = dist_list_sorted[:1][0]
        print('test template: %d, %s' % (test_id, templates[test_id]))
        closest_dist = closest_template_info[1]
        print('closest 1 train (dist %.3f): %d, %s' % (closest_dist, closest_template_info[0], templates[closest_template_info[0]]))
        print('')
        i = 1
        while True:
            template_info = dist_list_sorted[i:i+1][0]
            dist = template_info[1]
            if dist > closest_dist:
                break
            else:
                print('closest %d train (dist %.3f): %d, %s' % ((i+1), dist, template_info[0], templates[template_info[0]]))
                print('')
                i += 1

    

if __name__=='__main__':
    main()