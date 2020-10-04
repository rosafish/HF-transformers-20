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
    cached_train_features_file = '../cache/cached_cose_tmp' 
    save_trained_model_dir = "./cose_tmp_model/"
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


if __name__=='__main__':
    main()