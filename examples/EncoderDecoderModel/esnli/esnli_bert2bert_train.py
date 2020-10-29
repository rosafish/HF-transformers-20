import torch
import os
import sys
sys.path.append('./../../src')
import random
import numpy as np

from esnli_processor import EsnliProcessor
import logging as logger
from transformers import BertTokenizer
from esnli_processor import EsnliInputFeatures, esnli_examples_to_features, text_to_input_ids
from transformers import EncoderDecoderModel
from transformers import Trainer, TrainingArguments


def main():
    # set seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed()
    np.random.seed(0)

    # paths and params
    max_seq_len = 128
    train_data_path = './sanity-checks/esnli_train_1000.csv'
    cached_train_features_file = './cache/cached_sanity_checks_train_esnli1k'
    save_trained_model_dir = "./sanity-checks/esnli1k_train_trained_model/"
    # load dev data, because we are using dev data to find best model / number of steps to train for
    eval_data_path = './sanity-checks/esnli_dev_100.csv'
    
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
    if args.generate_expl_on_training_data:
        eval_examples = processor.get_train_examples(eval_data_path)
    else:
        eval_examples = processor.get_dev_examples(eval_data_path) 
    
    # convert dev examples to features
    eval_features = esnli_examples_to_features(eval_examples, max_seq_len, tokenizer)
    
    # Training
    #initialize Bert2Bert
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') 

    training_args = TrainingArguments(
        output_dir='./checkpoint-train-results',    # output directory
        per_device_train_batch_size=4,              # batch size per device during training
        weight_decay=0.01,                          # strength of weight decay
        logging_dir='./train-logs',                 # directory for storing logs
        do_train=True,
        # save best model
        evalute_during_training=True,
        esnli_evaluate_during_training=True,
        # modify the following for different sample size
        # num_train_epochs=3,                         # total # of training epochs
        max_steps = 20000,                          # overwrites num_train_epochs, this is here for few-sample learning specifically.
        logging_steps=5000,                         # I think it is good to set logging steps = saving steps = eval steps
        save_steps=5000,
        eval_steps=5000,
        overwrite_output_dir=True,
        warmup_steps=1000,                          # number of warmup steps for learning rate scheduler
    )

    trainer = Trainer(
        model=model,                                # the instantiated 🤗 Transformers model to be trained
        args=training_args,                         # training arguments, defined above
        train_dataset=train_features,               # training dataset
        eval_dataset=eval_features                 # evaluation dataset
    )

    trainer.train()

    # Save Model After Training
    output_dir = save_trained_model_dir
    cuda_id = "1" # since there's something running on 0 #TODO: is this line necessary?

    trainer.save_model(output_dir)

    
if __name__=='__main__':
    main()