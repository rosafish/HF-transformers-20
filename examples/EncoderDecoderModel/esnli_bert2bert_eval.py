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


def main(): 
    # cml args
    # $ python esnli_bert2bert_eval.py <PATH>
    output_dir = sys.argv[1]
    print("Directory that stores the model to evaluate:", output_dir)
    if not os.path.isdir(output_dir):
        print("The directory does not exist.")
        return
    
    # set seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed()
    np.random.seed(0)
    
    # paths and params
    dev_data_path = './sanity-checks/esnli_dev.csv'
    max_seq_len = 128
    cuda_id = "2" # since there's something running on the other ones
    
    # get examples
    processor = EsnliProcessor()
    dev_examples = processor.get_dev_examples(dev_data_path) 
    
    # convert examples to features
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dev_features = esnli_examples_to_features(dev_examples, max_seq_len, tokenizer)
    
    # Load a trained model and vocabulary that you have fine-tuned
    model = EncoderDecoderModel.from_pretrained(output_dir)
    device = torch.device("cuda:"+cuda_id)
    model.to(device)

    model.config.max_length=128
    model.config.decoder_start_token_id = 101
    model.config.eos_token_id = 102

    eval_args = TrainingArguments(
        output_dir='./checkpoint-eval-results',          # output directory
        per_device_eval_batch_size=1,   # batch size for evaluation
        do_eval = True,
        predict_from_generate=True,
    )

    evaluator = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=eval_args,                  # eval arguments, defined above
        eval_dataset=dev_features,            # evaluation dataset
    )

    # Evaluate
    evaluator.eval_esnli_write_output()  

if __name__=='__main__':
    main()