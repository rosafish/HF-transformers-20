import csv
import sys
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import TweetTokenizer

if __name__=='__main__':
    input_dir = sys.argv[1] # model input file directory
    output_csv = sys.argv[2] # model generated explanations text file
    expl_type = sys.argv[3]
    test_type = sys.argv[4]

    input_path = input_dir+test_type+'_test_text.csv'
    bleu_output_path = input_dir+test_type+'_test_bleu_by_temp.txt'

    input_expls_by_template = {}
    template_ids_by_line = []

    tknzr = TweetTokenizer()

    with open(input_csv, newline='') as f:
        reader = csv.reader(f)

        for (i, line) in enumerate(reader):

            template_id = line[1]
            template_ids_by_line.append(template_id)

            if i == 0:
                continue

            # if i > 5: 
            #     break
            
            if expl_type == 'pt':
                explanation_text = line[-1]
            elif expl_type == 'nl':
                explanation_text = line[-2]
            else:
                print('invalid expl type: ', expl_type)

            explanation_text = tknzr.tokenize(explanation_text)

            if template_id in input_expls_by_template:
                input_expls_by_template[template_id][0].append([explanation_text])
            else:
                input_expls_by_template[template_id]=[[[explanation_text]], []] # [[ref],[cand]]

    #print(input_expls_by_template)
    
    with open(output_csv, newline='') as f:
        reader = csv.reader(f)

        for (i, line) in enumerate(reader):
            if i == 0:
                continue

            # if i > 5: 
            #     break
            
            explanation_text = line[0]
            template_id = template_ids_by_line[i]

            explanation_text = tknzr.tokenize(explanation_text)
            
            input_expls_by_template[template_id][1].append(explanation_text)

    #print(input_expls_by_template)

    f = open(bleu_output_path, 'w+')
    for template_id in input_expls_by_template.keys():
        print('template id: ', template_id)
        bleu = corpus_bleu(input_expls_by_template[template_id][0], input_expls_by_template[template_id][1])
        rounded_blue = round(bleu, 5)*100
        print('BLEU: ', rounded_blue)
        f.write(template_id, ', ', rounded_blue, '\n')
    f.close()
            

            
