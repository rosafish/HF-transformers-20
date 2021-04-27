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
            # print("template_id: ", template_id)
            # print("bleu: ", bleu)

            bleu_list.append((template_id, bleu))
            all_test_templates_id.add(template_id)

    bleu_list_sorted = sort_tuple(bleu_list)

    worst_temp_info = bleu_list_sorted[:num_worst_temp]
    print(worst_temp_info)
    return [t[0] for t in worst_temp_info], all_test_templates_id


def replace_word_subtype2type(s):
    for k,v in var_type_subtypes.items():
        # print("key: ", k)
        # print("value: ", v)
        for subtype in v:
            s = s.replace(subtype, k)
    return s


def load_templates(templates_path):
    templates = []
    with open(templates_path, newline='') as f:
        reader = csv.reader(f)
        for (i, line) in enumerate(reader):
            if i == 0:
                continue

            p = replace_word_subtype2type(line[5])
            h = replace_word_subtype2type(line[6])
            
            templates.append(p+h)

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
    data_dir_name = 'before_new_setting'

    bleu_by_temp_path = '/net/scratch/zhouy1/randomness_experiment/%s/edm/%s_hans_seed%s_partition%s_train%s_%s/%s_test_bleu_by_temp.txt' % \
                        (data_dir_name, model, seed, partition, train_size, expl_type, test_type)

    # find worst five templates
    worst_templates_id, all_test_templates_id = find_worst_templates_id(bleu_by_temp_path ,5)
    print('worst 5 templates (id): ', worst_templates_id)

    all_templates_id = set([i for i in range(118)])
    all_train_templates_id = all_templates_id - all_test_templates_id

    templates_path = "/home/zhouy1/hans-forked/auto/templates_new.csv"
    templates = load_templates(templates_path)

    for test_id in worst_templates_id:
        jaccard_dist_list = []
        for train_id in all_train_templates_id:
            # if train_id != test_id:
            #     jaccard_dist = get_jaccard_dist(templates, test_id, train_id)
            #     jaccard_dist_list.append((train_id, jaccard_dist))
            
            jaccard_dist = get_jaccard_dist(templates, test_id, train_id)
            jaccard_dist_list.append((train_id, jaccard_dist))
            if train_id == test_id:
                print('train: %d, test: %d, jaccard: %f' % (train_id, test_id, jaccard_dist))
                print()

        jaccard_dist_list_sorted = sort_tuple(jaccard_dist_list)
        closest_template_info = jaccard_dist_list_sorted[:1][0]
        # print('closest_template_info: ', closest_template_info)
        print('test template: %d, %s' % (test_id, templates[test_id]))
        print('closest 1 train (dist %f): %d, %s' % (closest_template_info[1], closest_template_info[0], templates[closest_template_info[0]]))
        closest_template_info = jaccard_dist_list_sorted[1:2][0]
        print('closest 2 train (dist %f): %d, %s' % (closest_template_info[1], closest_template_info[0], templates[closest_template_info[0]]))
        closest_template_info = jaccard_dist_list_sorted[2:3][0]
        print('closest 3 train (dist %f): %d, %s' % (closest_template_info[1], closest_template_info[0], templates[closest_template_info[0]]))
        print('')


if __name__=='__main__':
    main()