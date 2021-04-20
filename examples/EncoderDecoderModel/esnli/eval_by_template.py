import csv
import sys
from nltk.translate.bleu_score import corpus_bleu

if __name__=='__main__':
    input_csv = sys.argv[1] # model input file
    output_csv = sys.argv[2] # model generated explanations text file
    expl_type = sys.argv[3]

    input_expls_by_template = {}

    with open(input_csv, newline='') as f:
        reader = csv.reader(f)

        for (i, line) in enumerate(reader):
            if i == 0:
                continue
            print(i)
            print(line)
            template_id = line[1]
            if expl_type == 'pt':
                explanation_text = line[-1]
            elif expl_type == 'nl':
                explanation_text = line[-2]
            else:
                print('invalid expl type: ', expl_type)

            if template_id in input_expls_by_template:
                input_expls_by_template[template_id].append(explanation_text)
            else:
                input_expls_by_template[template_id]=[explanation_text]

            if i > 5: 
                break

            print(input_expls_by_template)
        
        print(reader[1])
        print(reader[2])

            