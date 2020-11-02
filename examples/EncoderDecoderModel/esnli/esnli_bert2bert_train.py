import torch
import os
import sys
sys.path.append('./../../src')
import random
import argparse
import numpy as np
import logging as logger

from esnli_processor import EsnliProcessor
from transformers import BertTokenizer
from esnli_processor import EsnliInputFeatures, esnli_examples_to_features, text_to_input_ids
from transformers import EncoderDecoderModel
from transformers import Trainer, TrainingArguments


def main():
    parser = argparse.ArgumentParser(description='Path arguments')
    parser.add_argument('-data_dir', action="store", default="", type=str)
    parser.add_argument('-cached_train_features_file', action="store", default="", type=str)
    parser.add_argument('-save_trained_model_dir', action="store", default="", type=str)
    parser.add_argument('-train_epochs', action="store", default=3, type=int)
    parser.add_argument('-max_steps', action="store", default=-1, type=int)
    parser.add_argument('-hans', action="store_true", default=False) 
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed()
    np.random.seed(0)

    # paths and params
    max_seq_len = 128
    train_data_path = args.data_dir + 'esnli_train.csv'
    cached_train_features_file = args.cached_train_features_file
    save_trained_model_dir = args.save_trained_model_dir
    # load dev data, because we are using dev data to find best model / number of steps to train for
    eval_data_path = args.data_dir + 'esnli_dev.csv'
    
    # Get train examples
    processor = EsnliProcessor()
    train_examples = processor.get_train_examples(train_data_path) 

    # Convert train examples to features
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Cache training dataset features
    if os.path.exists(cached_train_features_file):
        logger.info("Loading features from cached file %s", cached_train_features_file)
        train_features = torch.load(cached_train_features_file)
    else:
        train_features = esnli_examples_to_features(train_examples, max_seq_len, tokenizer)
        logger.info("Saving training features into cached file %s", cached_train_features_file)
        torch.save(train_features, cached_train_features_file)

    # get dev examples
    eval_examples = processor.get_dev_examples(eval_data_path) 
    
    # convert dev examples to features
    eval_features = esnli_examples_to_features(eval_examples, max_seq_len, tokenizer)
    
    # Training
    #initialize Bert2Bert
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') 
    model.config.max_length=128
    model.config.decoder_start_token_id = 101
    model.config.eos_token_id = 102

    training_args = TrainingArguments(
        output_dir=save_trained_model_dir,          # output directory
        per_device_train_batch_size=4,              # batch size per device during training
        weight_decay=0.01,                          # strength of weight decay
        logging_dir='./train-logs',                 # directory for storing logs
        do_train=True,
        # save best model
        evaluate_during_training=True,
        esnli_evaluate_during_training=True,
        per_device_eval_batch_size=1,
        predict_from_generate=True,
        # modify the following for different sample size
        num_train_epochs=args.train_epochs,        # total # of training epochs
        max_steps=args.max_steps,                          # overwrites num_train_epochs, this is here for few-sample learning specifically.
        logging_steps=5000,                         
        overwrite_output_dir=True,
        warmup_steps=1000,                          # number of warmup steps for learning rate scheduler
    )

    trainer = Trainer(
        model=model,                                # the instantiated 🤗 Transformers model to be trained
        args=training_args,                         # training arguments, defined above
        train_dataset=train_features,               # training dataset
        eval_dataset=eval_features,                 # evaluation dataset
    )

    trainer.train()

    # Save Model After Training
    cuda_id = "1" # since there's something running on 0 #TODO: is this line necessary?
    trainer.save_model(save_trained_model_dir+"/last_model/") # no guarantee of that the last model is the best

    
if __name__=='__main__':
    main()