import csv
import matplotlib.pyplot as plt
from worst_test_templates import sort_tuple
import sys

def get_bleu_by_templates(it_bleu_by_temp_path, ot_bleu_by_temp_path):
    bleu_list = []
    all_test_templates_id = set()
    with open(it_bleu_by_temp_path, newline='') as f:
        reader = csv.reader(f)
        for (i, line) in enumerate(reader):

            template_id = eval(line[0])
            bleu = round(eval(line[1]), 5)

            bleu_list.append((template_id, bleu))
            all_test_templates_id.add(template_id)

    with open(ot_bleu_by_temp_path, newline='') as f:
        reader = csv.reader(f)
        for (i, line) in enumerate(reader):

            template_id = eval(line[0])
            bleu = round(eval(line[1]), 5)

            bleu_list.append((template_id, bleu))
            all_test_templates_id.add(template_id)

    assert len(all_test_templates_id) == 118

    bleu_list_ascend = sort_tuple(bleu_list) # ascending order

    return bleu_list_ascend


def main():
    seed = sys.argv[1]
    partition = sys.argv[2]
    expl_type = sys.argv[3]
    vocab_type = sys.argv[4] # ov or iv
    model = sys.argv[5]
    train_size = sys.argv[6]
    data_dir_name = 'before_new_setting'

    it_bleu_by_temp_path = '/net/scratch/zhouy1/randomness_experiment/%s/edm/%s_hans_seed%s_partition%s_train%s_%s/%s_test_bleu_by_temp.txt' % \
                        (data_dir_name, model, seed, partition, train_size, expl_type, vocab_type+'it')
    ot_bleu_by_temp_path = '/net/scratch/zhouy1/randomness_experiment/%s/edm/%s_hans_seed%s_partition%s_train%s_%s/%s_test_bleu_by_temp.txt' % \
                        (data_dir_name, model, seed, partition, train_size, expl_type, vocab_type+'ot')

    # list of tuples (temp_id, bleu)
    bleu_list_ascend = get_bleu_by_templates(it_bleu_by_temp_path, ot_bleu_by_temp_path) 

    # plot histogram
    bleu_list_histogram = [item[1] for item in bleu_list_ascend]
    plt.hist(bleu_list_histogram, bins=10)
    plt.savefig('test_histogram_%s_hans_seed%s_partition%s_train%s_%s_%s_%s.png' % (model, seed, partition, train_size, expl_type, vocab_type, data_dir_name))


if __name__=='__main__':
    main()
