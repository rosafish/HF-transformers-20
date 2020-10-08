# Turn the bert generated expl to be usable for evaluating the esnli expl classifier with the same processor
# Works for the e-snli data

import csv
import sys
sys.path.append('/data/rosa/my_github/expl-discourse/esnli/')
from sample_generated_expl import load_gold_expl, load_bert_expl
import argparse

def output_bertgen_to_original_format(output_csv_path, bert_expl_data, gold_expl_path):
    gold_expl_headers = []
    with open(gold_expl_path) as input_file:
        reader = csv.reader(input_file, delimiter=',')
        for (i, line) in enumerate(reader):
                if i == 0:
                    gold_expl_headers = line
                    break

    fieldnames = ['pairID', 'gold_label', 'Sentence1', 'Sentence2', 'bert_expl']
    fieldnames.extend(['dummy']*(len(gold_expl_headers)-5))

    with open(output_csv_path, mode='w') as output_file:
        writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fieldnames)
        
        with open(gold_expl_path) as input_file:
            reader = csv.reader(input_file, delimiter=',')
            for (i, line) in enumerate(reader):
                if i == 0:
                    continue
                line[4] = bert_expl_data['pred_expl'][i-1]
                for j in range(5, len(fieldnames)):
                    line[j] = ""
                writer.writerow(line)

def main():
    parser = argparse.ArgumentParser(description='Path arguments')
    parser.add_argument('-gold_expl_csv_path', action="store", default='/data/rosa/data/esnli/esnli_dev.csv', type=str)
    parser.add_argument('-bert_expl_csv_path', action="store", default="", type=str)
    parser.add_argument('-output_csv_path', action="store", default="", type=str)
    parser.add_argument('-hans', action="store_true", default=False)
    args = parser.parse_args()
    
    # hans
    gold_expl_path = args.gold_expl_csv_path if not args.hans else '/data/rosa/data/hans/in_esnli_format/esnli_dev.csv'
    
    print('gold expl path: ', gold_expl_path)
    
    bert_expl_data = load_bert_expl(args.bert_expl_csv_path)
    output_bertgen_to_original_format(args.output_csv_path, bert_expl_data, gold_expl_path)
    
if __name__=="__main__":
    main()