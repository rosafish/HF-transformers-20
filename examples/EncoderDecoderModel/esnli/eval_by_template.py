import csv
import sys
from nltk.translate.bleu_score import corpus_bleu

if __name__=='__main__':
    input_csv = sys.argv[1] # model input file
    output_csv = sys.argv[2] # model generated explanations text file
    expl_type = sys.argv[3]

    input_expls_by_template = {}
    template_ids_by_line = []

    with open(input_csv, newline='') as f:
        reader = csv.reader(f)

        for (i, line) in enumerate(reader):

            template_id = line[1]
            template_ids_by_line.append(template_id)

            if i == 0:
                continue

            if i > 5: 
                break
            
            if expl_type == 'pt':
                explanation_text = line[-1]
            elif expl_type == 'nl':
                explanation_text = line[-2]
            else:
                print('invalid expl type: ', expl_type)

            #TODO: do i need to tokenize explanations into tokens first?

            if template_id in input_expls_by_template:
                input_expls_by_template[template_id][0].append(explanation_text)
            else:
                input_expls_by_template[template_id]=[[explanation_text], []] # [[ref],[cand]]

    print(input_expls_by_template)
    
    with open(output_csv, newline='') as f:
        reader = csv.reader(f)

        for (i, line) in enumerate(reader):
            if i == 0:
                continue

            if i > 5: 
                break
            
            explanation_text = line[0]
            template_id = template_ids_by_line[i]

            #TODO: do i need to tokenize explanations into tokens first?
            
            input_expls_by_template[template_id][1].append(explanation_text)

    print(input_expls_by_template)
            

            