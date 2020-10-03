# convert generated explanations from their embeddings form to texts
# works for the e-snli data

import csv
from transformers import BertTokenizer
from nltk.translate.bleu_score import corpus_bleu

import sys
sys.path.append('/data/rosa/my_github/misinformation/code/')
from myTools import write_csv

import argparse


def remove_special_tokens(token_list):    
    sep_index = token_list.index(102) if 102 in token_list else -1
    cls_index = token_list.index(101) if 101 in token_list else -1
    
    result = []
    # remove [sep] and [pad]
    if sep_index == -1:
        result = token_list
    else:
        result = token_list[:sep_index]
    
    # remove [cls]
    if cls_index == -1:
        return result
    else:
        return result[1:]
    
    
def compute_embedding_bleu(embedding_csv_path):
    ref_123 = []
    ref_12 = []
    ref_13 = []
    ref_23 = []
    cand_1 = []
    cand_2 = []
    cand_3 = []
    cand_bert = []
    with open(embedding_csv_path, newline='') as f:
        reader = csv.reader(f)
        for (i, line) in enumerate(reader):
            if i == 0:
                continue
            pred_expl = eval(line[0])
            gold_expl_1 = eval(line[1])
            gold_expl_2 = ""
            gold_expl_3 = ""
            if line[2] and line[3]:
                gold_expl_2 = eval(line[2])
                gold_expl_3 = eval(line[3])
            # process the explanations before passing to compute bleu scores: 
            # get rid of the CLS, SEP, can PAD tokens - tokens with id 101, 102, and 0
            pred_expl = remove_special_tokens(pred_expl)
            gold_expl_1 = remove_special_tokens(gold_expl_1)
            if line[2] and line[3]:
                gold_expl_2 = remove_special_tokens(gold_expl_2)
                gold_expl_3 = remove_special_tokens(gold_expl_3)
            
            ref_123.append([gold_expl_1, gold_expl_2, gold_expl_3])
            ref_12.append([gold_expl_1, gold_expl_2])
            ref_13.append([gold_expl_1, gold_expl_3])
            ref_23.append([gold_expl_2, gold_expl_3])
            cand_1.append(gold_expl_1)
            cand_2.append(gold_expl_2)
            cand_3.append(gold_expl_3)
            cand_bert.append(pred_expl)
    print('gold expl1 with respect to gold 2,3: ', corpus_bleu(ref_23, cand_1))
    print('gold expl2 with respect to gold 1,3: ', corpus_bleu(ref_13, cand_2))
    print('gold expl3 with respect to gold 1,2: ', corpus_bleu(ref_12, cand_3))
    
    print('bert expl with respect to gold 2,3: ', corpus_bleu(ref_23, cand_bert))
    print('bert expl with respect to gold 1,3: ', corpus_bleu(ref_13, cand_bert))
    print('bert expl with respect to gold 1,2: ', corpus_bleu(ref_12, cand_bert))
    
    print('bert expl with respect to gold 1,2,3: ', corpus_bleu(ref_123, cand_bert))

            
def convert_embedding2text(embedding_csv_path, text_csv_path, tokenizer):
    # works for e-snli data
    output_csv_header = []
    output_csv_rows = []
    with open(embedding_csv_path, newline='') as f:
        reader = csv.reader(f)
        for (i, line) in enumerate(reader):
            if i == 0:
                output_csv_header = line
                continue
            pred_expl = eval(line[0])
            gold_expl_1 = eval(line[1])
            gold_expl_2 = ""
            gold_expl_3 = ""
            if line[2] and line[3]:
                gold_expl_2 = eval(line[2])
                gold_expl_3 = eval(line[3])
            # process the explanations before passing to compute bleu scores: 
            # get rid of the CLS, SEP, can PAD tokens - tokens with id 101, 102, and 0
            pred_expl = remove_special_tokens(pred_expl)
            gold_expl_1 = remove_special_tokens(gold_expl_1)
            if line[2] and line[3]:
                gold_expl_2 = remove_special_tokens(gold_expl_2)
                gold_expl_3 = remove_special_tokens(gold_expl_3)
            # convert embedding to text
            pred_expl = tokenizer.decode(pred_expl)
            gold_expl_1 = tokenizer.decode(gold_expl_1)
            if line[2] and line[3]:
                gold_expl_2 = tokenizer.decode(gold_expl_2)
                gold_expl_3 = tokenizer.decode(gold_expl_3)
            output_csv_rows.append([pred_expl, gold_expl_1, gold_expl_2, gold_expl_3])
    write_csv(text_csv_path, output_csv_rows, output_csv_header)


def main():
    parser = argparse.ArgumentParser(description='Path arguments')
    parser.add_argument('-embedding_csv_path', action="store", default="", type=str)
    parser.add_argument('-text_csv_path', action="store", default="", type=str)
    args = parser.parse_args()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print("compute bleu scores on embedding first...")
    compute_embedding_bleu(args.embedding_csv_path)
    
    print("converting embedding to text...")
    convert_embedding2text(args.embedding_csv_path, args.text_csv_path, tokenizer)

if __name__=='__main__':
    main()
