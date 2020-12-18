import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)
#%%
class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

#%%
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

#%%
def seq_cls_convert_examples_to_features(args, examples, tokenizer, max_length):
    processor = DataProcessor(args)
    label_list = processor.get_labels()

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example):
        return label_map[example.label]


    labels = [label_from_example(example) for example in examples]
            
    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = [0] * len(inputs["input_ids"])  # For xlm-roberta

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    # for i, example in enumerate(examples[:5]):
    #     logger.info("*** Example ***")
    #     logger.info("guid: {}".format(example.guid))
    #     logger.info("input_ids: {}".format(" ".join([str(x) for x in features[i].input_ids])))
    #     logger.info("attention_mask: {}".format(" ".join([str(x) for x in features[i].attention_mask])))
    #     logger.info("token_type_ids: {}".format(" ".join([str(x) for x in features[i].token_type_ids])))
    #     logger.info("label: {}".format(features[i].label))

    return features
#%%
class DataProcessor(object):
    """Processor for the NSMC data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["0", "1", "2", "3", "4"]

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[0:]):

            guid = "%s-%s" % (set_type, i)
            text_a = line[0:-1]
            label = line[-1]     
            if i % 10000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples( self._read_file(os.path.join(self.args.data_dir, file_to_read)), mode )
#%%
def seq_cls_load_and_cache_examples(args, tokenizer, mode):
    processor = DataProcessor(args)
    # Load data features from cache or dataset file
    if mode == "train":
        cached_feature_folder = os.path.join(args.feature_dir, mode, args.train_file)
    elif mode == "test":
        cached_feature_folder = os.path.join(args.feature_dir, mode, args.test_file)
    elif mode == "dev":
        cached_feature_folder = os.path.join(args.feature_dir, mode, args.dev_file)
        
        
    if not os.path.exists(cached_feature_folder):
        os.makedirs(cached_feature_folder)
        
        
    cached_features_file = os.path.join(
        cached_feature_folder,
        "cached_{}_{}_{}".format( list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_len), mode ),
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.feature_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise ValueError("For mode, only train, dev, test is avaiable")
        features = seq_cls_convert_examples_to_features( args, examples, tokenizer, max_length=args.max_seq_len )
        # logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset
