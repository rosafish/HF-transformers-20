import sys
import csv

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

def find_worst_templates_id(bleu_by_temp_path, num_worst_temp):
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

def get_jaccard_dist(templates, test_id, train_id):
    s1 = templates[test_id].lower()
    s2 = templates[train_id].lower()

    tknzr = TweetTokenizer()

    s1 = set(tknzr.tokenize(s1))
    s2 = set(tknzr.tokenize(s2))
    
    # Find the intersection of words list of doc1 & doc2
    intersection = s1.intersection(s2)

    # Find the union of words list of doc1 & doc2
    union = s1.union(s2)
        
    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)

def main():
    seed = sys.argv[1]
    partition = sys.argv[2]
    expl_type = sys.argv[3]
    test_type = sys.argv[4]
    model = sys.argv[5]
    train_size = sys.argv[6]
    input_type = sys.argv[7] # p, h, or p+h
    data_dir_name = 'before_new_setting'

    bleu_by_temp_path = '/net/scratch/zhouy1/randomness_experiment/%s/edm/%s_hans_seed%s_partition%s_train%s_%s/%s_test_bleu_by_temp.txt' % \
                        (data_dir_name, model, seed, partition, train_size, expl_type, test_type)

    # find worst three templates
    worst_templates_id, all_test_templates_id = find_worst_templates_id(bleu_by_temp_path ,3)
    print('worst 3 templates (id): ', worst_templates_id)

    all_templates_id = set([i for i in range(118)])
    all_train_templates_id = all_templates_id - all_test_templates_id

    templates_path = "/home/zhouy1/hans-forked/auto/templates_new.csv"
    templates = load_templates(templates_path, input_type)

    for test_id in worst_templates_id:
        jaccard_dist_list = []
        for train_id in all_train_templates_id:
            if train_id != test_id:
                jaccard_dist = get_jaccard_dist(templates, test_id, train_id)
                jaccard_dist_list.append((train_id, jaccard_dist))

        jaccard_dist_list_sorted = sort_tuple(jaccard_dist_list)
        closest_template_info = jaccard_dist_list_sorted[-1:][0]
        print('test template: %d, %s' % (test_id, templates[test_id]))
        closest_dist = closest_template_info[1]
        print('closest 1 train (dist %f): %d, %s' % (closest_dist, closest_template_info[0], templates[closest_template_info[0]]))
        print('')
        i = 1
        while True:
            template_info = jaccard_dist_list_sorted[-1-i:0-i][0]
            dist = template_info[1]
            if dist < closest_dist:
                break
            else:
                print('closest %d train (dist %f): %d, %s' % ((i+1), dist, template_info[0], templates[template_info[0]]))
                print('')
                i += 1

if __name__=='__main__':
    main()