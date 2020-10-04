# This file should be called after running the ./cose_bert2bert_train.py file, 
# which trains an encoder-decoder model for seq2seq tasks.
import torch
import os
import random
import numpy as np
import argparse
# import logging as logger
# import sys
# sys.path.append('./../../src')

from transformers import DataProcessor, InputExample, BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
from cose_processor import CoseProcessor, CoseInputFeatures, cose_examples_to_features


def main():
    # cml args
    parser = argparse.ArgumentParser(description='Path arguments')
    parser.add_argument('-model_dir', action="store", default="./cose_trained_model/", type=str)
    parser.add_argument('-eval_qa_data_path', action="store", default='/data/rosa/data/cos-e/data/v1.0/dev_rand_split.jsonl', type=str)
    parser.add_argument('-eval_expl_data_path', action="store", default='/data/rosa/data/cos-e/data/v1.0/cose_dev_v1.0.jsonl', type=str)
    parser.add_argument('-generate_expl_on_training_data', action="store_true", default=False)
    args = parser.parse_args()

    # check if model directory exist
    output_dir = args.model_dir
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
    print('Eval qa data path: ', args.eval_qa_data_path)
    print('Eval expl data path: ', args.eval_expl_data_path)
    max_seq_len = 128
    cuda_id = "2" # since there's something running on the other ones
    
    # get examples
    processor = CoseProcessor()
    if args.generate_expl_on_training_data:
        eval_examples = processor.get_train_examples(args.eval_qa_data_path, args.eval_expl_data_path)
    else:
        eval_examples = processor.get_dev_examples(args.eval_qa_data_path, args.eval_expl_data_path) 
    
    # convert examples to features
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    eval_features = cose_examples_to_features(eval_examples, max_seq_len, tokenizer)
    
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
        eval_dataset=eval_features,            # evaluation dataset
    )

    # Evaluate
    evaluator.eval_cose_write_output(tokenizer=tokenizer)


if __name__=='__main__':
    main()


