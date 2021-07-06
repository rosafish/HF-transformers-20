# This file should be called after running the ./esnli_bert2bert_train.py file, which trains an encoder-decoder model for seq2seq tasks.
import torch
import os
import sys
sys.path.append('./../../src')
import random
import numpy as np

from transformers import DataProcessor, InputExample
import csv

import logging as logger
from transformers import BertTokenizer

from transformers import EncoderDecoderModel
from transformers import Trainer, TrainingArguments

from esnli_processor import EsnliProcessor
from esnli_processor import EsnliInputFeatures, esnli_examples_to_features, text_to_input_ids

import argparse


def main(): 
    # cml args
    # $ python esnli_bert2bert_eval.py -model_dir <PATH> -eval_data_path <PATH>
    parser = argparse.ArgumentParser(description='Path arguments')
    parser.add_argument('-model_dir', action="store", default="./esnli_task_trained_model_and_results/esnli/esnli_train_trained_model_copy/", type=str)
    parser.add_argument('-eval_data_path', action="store", default='./sanity-checks/esnli_dev.csv', type=str)
    parser.add_argument('-eval_results_dir', action="store", default="", type=str)
    parser.add_argument('-hans_original_eval_data', action="store_true", default=False)  
    parser.add_argument('-generate_expl_on_training_data', action="store_true", default=False)
    args = parser.parse_args()
    
    # check if model directory exist
    output_dir = args.model_dir
    print("Directory that stores the model to evaluate:", output_dir)
    if not os.path.isdir(output_dir):
        raise ValueError("The directory does not exist.")
    
    # set seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed()
    np.random.seed(0)
    
    # paths and params
    if args.hans_original_eval_data:
        eval_data_path = "/data/rosa/data/hans/in_esnli_format/esnli_dev.csv"
    else:
        eval_data_path = args.eval_data_path
    print('Eval data path: ', eval_data_path)
    max_seq_len = 128
    cuda_id = "0" # since there's something running on the other ones
    
    # get examples
    processor = EsnliProcessor()
    if args.generate_expl_on_training_data:
        eval_examples = processor.get_train_examples(eval_data_path)
    else:
        eval_examples = processor.get_dev_examples(eval_data_path) 
    
    # convert examples to features
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    eval_features = esnli_examples_to_features(eval_examples, max_seq_len, tokenizer)
    
    # Load a trained model and vocabulary that you have fine-tuned
    model = EncoderDecoderModel.from_pretrained(output_dir)
    device = torch.device("cuda:"+cuda_id)
    model.to(device)

    model.config.max_length=128
    model.config.decoder_start_token_id = 101
    model.config.eos_token_id = 102

    eval_args = TrainingArguments(
        output_dir=args.eval_results_dir,          # output directory
        per_device_eval_batch_size=4,              # batch size for evaluation
        do_eval = True,
        predict_from_generate=True,
    )

    evaluator = Trainer(
        model=model,                               # the instantiated HuggingFace Transformers model to be trained
        args=eval_args,                            # eval arguments, defined above
        eval_dataset=eval_features,                # evaluation dataset
    )

    # Evaluate
    evaluator.eval_esnli_write_output()  

if __name__=='__main__':
    main()
