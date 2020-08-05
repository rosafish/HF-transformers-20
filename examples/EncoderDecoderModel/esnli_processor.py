import logging
import os

logger = logging.getLogger(__name__)

from transformers import DataProcessor, InputExample
import csv

class EsnliProcessor(DataProcessor):

    def get_train_examples(self, data_path):
        """See base class."""
        examples = []
        with open(data_path, newline='') as f:
            reader = csv.reader(f)
            for (i, line) in enumerate(reader):
                if i == 0:
                    continue
                guid = "%s-%s" % ("train", i)
                label = line[1]
                premise = line[2]
                hypothesis = line[3]
                text_a = premise + " [SEP] " + hypothesis # p + [SEP] + h
                text_b = line[4] # expl
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
    def get_dev_examples(self, data_path):
        examples = []
        with open(data_path, newline='') as f:
            reader = csv.reader(f)
            for (i, line) in enumerate(reader):
                if i == 0:
                    continue
                guid = "%s-%s" % ("dev", i)
                label = line[1]
                premise = line[2]
                hypothesis = line[3]
                text_a = premise + " [SEP] " + hypothesis # p + [SEP] + h
                text_b = line[4] # expl 1
                text_c = line[9] # expl 2
                text_d = line[14] # expl 3
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str) \
                and isinstance(text_c, str) and isinstance(text_d, str)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, \
                                             label=label, text_c=text_c, text_d=text_d))
        return examples

def esnli_examples_to_features(examples, max_seq_len, tokenizer, cls_token='[CLS]', sep_token='[SEP]', 
                               pad_token=0, mask_padding_with_zero=True):
    """
        Does not support token_type_id, because the EncoderDecoderModel does not. Therefore, the premise
        and hypothesis is separated by a [SEP], but no token_type_id is there to tell this difference.
    """
    
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        input_ids, input_mask = text_to_input_ids(example.text_a, max_seq_len, tokenizer)
        decoder_input_ids, dummy = text_to_input_ids(example.text_b, max_seq_len, tokenizer)
        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(decoder_input_ids) == max_seq_len
        
        expl2_ids = None
        expl3_ids = None
        if example.text_c != None and example.text_d != None:
            expl2_ids, dummy = text_to_input_ids(example.text_c, max_seq_len, tokenizer)
            expl3_ids, dummy = text_to_input_ids(example.text_d, max_seq_len, tokenizer)
            assert len(expl2_ids) == max_seq_len
            assert len(expl3_ids) == max_seq_len
        

        features.append(EsnliInputFeatures(input_ids=input_ids,
                                          attention_mask=input_mask,
                                          decoder_input_ids=decoder_input_ids,
                                          expl2_ids=expl2_ids,
                                          expl3_ids=expl3_ids))
    return features

def text_to_input_ids(text, max_seq_len, tokenizer, cls_token='[CLS]', sep_token='[SEP]', 
                      pad_token=0, mask_padding_with_zero=True):
    tokens = tokenizer.tokenize(text)
    
    # truncate to max_length - 2 if needed, 
    # the -2 accounts for cls_token and sep_token that are going to be added.
    tokens = tokens[:max_seq_len-2]
    
    tokens = [cls_token] + tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_len - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    
    return input_ids, input_mask

class EsnliInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, decoder_input_ids, expl2_ids, expl3_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.decoder_input_ids = decoder_input_ids
        self.labels = decoder_input_ids #expl1
        self.expl2 = expl2_ids #expl2
        self.expl3 = expl3_ids #expl3