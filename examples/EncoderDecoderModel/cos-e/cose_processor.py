import logging
import os

logger = logging.getLogger(__name__)

from transformers import DataProcessor, InputExample
import json
import sys
sys.path.append('/data/rosa/HF-transformers-20/examples/EncoderDecoderModel/esnli')
from esnli_processor import text_to_input_ids

class CoseProcessor(DataProcessor):

    def get_train_examples(self, qa_data_path, expl_data_path):
        """
        Input:
        qa_data_path := path to CQA data containing id, answer key, answer choices, question
        expl_data_path := path to CoS-E data containing id, expl

        Return a list of examples. 
        Each example consists of:
        - guid
        - label: answer key
        - text_a: q, c0, c1, or c2? commonsense says
        - text_b: expl (open-ended)
        """
        examples = []
        qa_info = self._get_qa_info(qa_data_path) # list of (id, answer key, c0, c1, c2, question)
        id_expl_dict = self._get_expl_dict(expl_data_path) # dict <id, expl>

        # combine qa info with expl to create examples
        if len(qa_info) != len(id_expl_dict.keys()):
            raise ValueError("CQA data and CoS-E data have different number of entries")
        for i in range(len(qa_info)):
            # check if ids are matched 
            guid = qa_info[i][0]
            label = qa_info[i][1]
            text_a = "%s, %s, %s, or %s? commonsense says" % \
                    (qa_info[i][5], qa_info[i][2], qa_info[i][3], qa_info[i][4])

            if guid in id_expl_dict:
                text_b = id_expl_dict[guid] # explanation
            else:
                raise ValueError("Id %s in CQA does not exist in CoS-E data." % guid)

            # check types
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, label=label, text_a=text_a, text_b=text_b))
        return examples

    def get_dev_examples(self, qa_data_path, expl_data_path):
        # cos-e train and dev datasets have the same format
        return self.get_train_examples(qa_data_path, expl_data_path)

    def _get_qa_info(self, qa_data_path):
        # returns a list of tuples (id, answer key, c0, c1, c2, question)
        results = []
        with open(qa_data_path, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)
            qa_tuple = (result['id'], result['answerKey'], 
                      result['question']['choices'][0]['text'],
                      result['question']['choices'][1]['text'],
                      result['question']['choices'][2]['text'], 
                      result['question']['stem'])
            results.append(qa_tuple)
        return results

    def _get_expl_dict(self, expl_data_path):
        # turns the json file into a dict <k,v> = <id, expl>
        id_expl_dicts = dict()
        with open(expl_data_path, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)
            expl_id = result['id']
            expl = result['explanation']['open-ended']
            id_expl_dicts[expl_id] = expl
        return id_expl_dicts

def cose_examples_to_features(examples, max_seq_len, tokenizer, cls_token='[CLS]', sep_token='[SEP]', 
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

        features.append(CoseInputFeatures(input_ids=input_ids,
                                          attention_mask=input_mask,
                                          decoder_input_ids=decoder_input_ids))
    return features

class CoseInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, decoder_input_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.decoder_input_ids = decoder_input_ids #expl1
        self.labels = decoder_input_ids #expl1