import torch
import os
import sys
sys.path.append('./../../src')
import random
import numpy as np

from cose_processor import CoseProcessor
import logging as logger
from transformers import BertTokenizer
from cose_processor import CoseInputFeatures, cose_examples_to_features
from transformers import EncoderDecoderModel
from transformers import Trainer, TrainingArguments


def main():
    # set seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed()
    np.random.seed(0)

    # paths and params
    train_qa_data_path = '/data/rosa/data/cos-e/data/v1.0/train_rand_split.jsonl'
    train_expl_data_path = '/data/rosa/data/cos-e/data/v1.0/cose_train_v1.0.jsonl'
    cached_train_features_file = '../cache/cached_train_cose' 
    save_trained_model_dir = "./cose_trained_model/"
    max_seq_len = 128

    # Get examples
    processor = CoseProcessor()
    train_examples = processor.get_train_examples(train_qa_data_path, train_expl_data_path) 

    # Convert examples to features 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Cache training dataset features
    if os.path.exists(cached_train_features_file):
        logger.info("Loading features from cached file %s", cached_train_features_file)
        train_features = torch.load(cached_train_features_file)
    else:
        train_features = cose_examples_to_features(train_examples, max_seq_len, tokenizer)
        logger.info("Saving training features into cached file %s", cached_train_features_file)
        torch.save(train_features, cached_train_features_file)

    # Training
    #initialize Bert2Bert
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') 

    training_args = TrainingArguments(
        output_dir='./checkpoint-train-results',    # output directory
        per_device_train_batch_size=4,              # batch size per device during training
        weight_decay=0.01,                          # strength of weight decay
        logging_dir='./train-logs',                 # directory for storing logs
        do_train=True,
        num_train_epochs=10,                        # total number of training epochs
        logging_steps=5000,
        save_steps=5000,
        overwrite_output_dir=True,
        warmup_steps=1000,                          # number of warmup steps for learning rate scheduler
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_features,        # training dataset
        eval_dataset=train_features          # evaluation dataset, does not matter here b/c I don't eval during training
    )

    trainer.train()

    # Save Model After Training
    output_dir = save_trained_model_dir
    cuda_id = "1" # since there's something running on 0

    trainer.save_model(output_dir)


if __name__=='__main__':
    main()