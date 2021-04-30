import csv
import matplotlib.pyplot as plt
from worst_test_templates import sort_tuple, get_jaccard_dist, load_templates
import seaborn as sns
import sys
import numpy as np

def get_bleu_by_templates(bleu_by_temp_path):
    bleu_list = []
    all_test_templates_id = set()
    with open(bleu_by_temp_path, newline='') as f:
        reader = csv.reader(f)
        for (i, line) in enumerate(reader):

            template_id = eval(line[0])
            bleu = round(eval(line[1]), 5)

            bleu_list.append((template_id, bleu))
            all_test_templates_id.add(template_id)

    bleu_list_ascend = sort_tuple(bleu_list) # ascending order

    return bleu_list_ascend


def main():
    seed = sys.argv[1]
    partition = sys.argv[2]
    expl_type = sys.argv[3]
    test_type = sys.argv[4] 
    model = sys.argv[5]
    train_size = sys.argv[6]
    input_type = sys.argv[7]
    data_dir_name = 'before_new_setting'

    bleu_by_temp_path = '/net/scratch/zhouy1/randomness_experiment/%s/edm/%s_hans_seed%s_partition%s_train%s_%s/%s_test_bleu_by_temp.txt' % \
                        (data_dir_name, model, seed, partition, train_size, expl_type, test_type)

    # list of tuples (temp_id, bleu)
    bleu_list_ascend = get_bleu_by_templates(bleu_by_temp_path) 
    print(bleu_list_ascend)
    ids_ascend = [item[0] for item in bleu_list_ascend]
    num_templates = len(ids_ascend)
    print('num_templates: ', num_templates)

    plot histogram
    bleu_list_histogram = [item[1] for item in bleu_list_ascend]
    plt.xlim(xmin=0, xmax = 100)
    plt.hist(bleu_list_histogram, bins=10)
    plt.title('Distribution of Templates by BLEU Scores (%s)' % test_type)
    plt.xlabel('BLEU Scores')
    plt.ylabel('Number of Templates')
    plt.savefig('histogram_%s_hans_seed%s_partition%s_train%s_%s_%s_%s.png' % (model, seed, partition, train_size, expl_type, test_type, data_dir_name))

    # templates_path = "/home/zhouy1/hans-forked/auto/templates_new.csv"
    # templates = load_templates(templates_path, input_type)
    # jaccard_sim_matrix = np.zeros((num_templates, num_templates))
    # for i in range(num_templates):
    #     for j in range(num_templates):
    #         i_id = ids_ascend[i]
    #         j_id = ids_ascend[j]
    #         jaccard_sim_matrix[i,j] = get_jaccard_dist(templates, i_id, j_id)
    # ax = sns.heatmap(jaccard_sim_matrix, linewidth=0.5)
    # plt.savefig('test_jaccard_heatmap_%s_hans_seed%s_partition%s_train%s_%s_%s_%s_%s.png' % (model, seed, partition, train_size, expl_type, test_type, input_type, data_dir_name))

if __name__=='__main__':
    main()
