# Turn the bert generated expl to be usable for evaluating the esnli expl classifier with the same processor

import csv
import sys
sys.path.append('/data/rosa/my_github/expl-discourse/esnli/')
from sample_generated_expl import load_gold_expl, load_bert_expl
import argparse

def output_bertgen_to_original_format(output_csv_path, bert_expl_data, gold_expl_path):
    '''
    esnli_dev.csv original headers:
        pairID,gold_label,Sentence1,Sentence2,
        Explanation_1,Sentence1_marked_1,Sentence2_marked_1,Sentence1_Highlighted_1,Sentence2_Highlighted_1,
        Explanation_2,Sentence1_marked_2,Sentence2_marked_2,Sentence1_Highlighted_2,Sentence2_Highlighted_2,
        Explanation_3,Sentence1_marked_3,Sentence2_marked_3,Sentence1_Highlighted_3,Sentence2_Highlighted_3
        
    I want to have bert generated expl be at the Explanation_1 place, and keep pairID,gold_label,Sentence1,Sentence2.
    The rest can still have the headers/headers place, but the entries will be empty strings in their columns.
    '''
    fieldnames = ['pairID', 'gold_label', 'Sentence1', 'Sentence2', \
                  'bert_expl', 'dummy', 'dummy', 'dummy', 'dummy', \
                  'dummy', 'dummy', 'dummy', 'dummy', 'dummy', \
                  'dummy', 'dummy', 'dummy', 'dummy', 'dummy']

    with open(output_csv_path, mode='w') as output_file:
        writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fieldnames)
        
        with open(gold_expl_path) as input_file:
            reader = csv.reader(input_file, delimiter=',')
            for (i, line) in enumerate(reader):
                if i == 0:
                    continue
                line[4] = bert_expl_data['pred_expl'][i-1]
                for j in range(5, 19):
                    line[j] = ""
                writer.writerow(line)

def main():
    parser = argparse.ArgumentParser(description='Path arguments')
    parser.add_argument('-bert_expl_csv_path', action="store", default="", type=str)
    parser.add_argument('-output_csv_path', action="store", default="", type=str)
    args = parser.parse_args()
    
    gold_expl_path = '/data/rosa/data/esnli/esnli_dev.csv'
    bert_expl_data = load_bert_expl(args.bert_expl_csv_path)
    
    output_bertgen_to_original_format(args.output_csv_path, bert_expl_data, gold_expl_path)
    
if __name__=="__main__":
    main()