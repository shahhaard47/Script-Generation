# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

import sys
import csv
import glob
import json
import logging
from collections import defaultdict
import os
from typing import List
import re
import torch

import tqdm

from transformers import PreTrainedTokenizer

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, title, script, genres):
        """Constructs a InputExample.
        Args:
            title: Title of the movie.
            script: Full script of the movie.
            genres: list. List of array containing binary values corresponding to the movie's genres
        """
        self.title = title
        self.script = script
        self.genres = genres


class InputFeatures(object):
    def __init__(self, example_id, choices_features):
        self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for input_ids, input_mask, segment_ids in choices_features
        ]


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != "label":
            raise ValueError("For training, the input file must contain a label column.")


        # examples = [
        #     InputExample(
        #         example_id=line[2],
        #         question=line[5],  # in the swag dataset, the
        #         # common beginning of each
        #         # choice is stored in "sent2".
        #         contexts=[line[4], line[4], line[4], line[4]],
        #         endings=[line[7], line[8], line[9], line[10]],
        #         label=line[11],
        #     )
        #     for line in lines[1:]  # we skip the line with the column names
        # ]

        return examples

class ScriptProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "genre.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        genre_dict = defaultdict(str)
        examples = []
        acceptable_genres = []
        for i, line in enumerate(lines):
        	if i == 0:
        		# print(line[4:32])
        		for j,val in enumerate(line[5:32]):
        			genre_dict[j] = val
        	else:
        		if not line[1].isspace():
        			examples.append(InputExample(
        				title = line[0],
        				script = re.sub(' +', ' ', str(line[1])),
        				genres = line[5:32]
        				))
        return examples, genre_dict




def convert_examples_to_features(
    examples: List[InputExample],
    genre_dict: dict,
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
    model_type = 'gpt2',
    acceptable_genres = None,
    max_tokens = None
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    # label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    max_len = 0
    count = 0
    outer_max_tokens = max_tokens
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        genre_tags = []
        # print(example.genres)
        # print(genre_dict)

        for id_a,a in enumerate(example.genres):
        	if int(a) == 1:
        		# print(a)
        		if '<' + genre_dict[int(id_a)] + '>' not in genre_tags:
        			if genre_dict[int(id_a)] in acceptable_genres:
        				genre_tags.append('<' + genre_dict[int(id_a)] + '>')
        if len(genre_tags) == 0:
        	continue

        text = example.script
        # print(example.title)
        inputs = tokenizer.encode_plus(example.script)
        # print(inputs)\
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        if len(input_ids) == 0:
        	# print(example.title)
        	# sdfsdf
        	continue
        # print(input_ids)
        index = 0
        overlap = 20
        split_inputs = []
        split_token_type_ids = []
        # print(max_tokens)
        if not outer_max_tokens:
        	max_tokens = len(input_ids)
        max_tokens = min(max_tokens, len(input_ids))
        # print(max_tokens)
        # print(len(input_ids))
        while index < max_tokens:
        	# print(index, min(index + (max_length - len(genre_tags)) , max_tokens))
        	split_inputs.append(torch.tensor(input_ids[index: min(index + (max_length - len(genre_tags)) , max_tokens)]))
        	index += ((max_length - len(genre_tags)) - overlap)
        index = 0
        while index < max_tokens:
        	split_token_type_ids.append(torch.tensor(token_type_ids[index: min(index + (max_length - len(genre_tags)) , max_tokens)]))
        	index += ((max_length - len(genre_tags)) - overlap)

        # print(split_inputs)

        split_inputs = [torch.cat((torch.tensor(tokenizer.encode(genre_tags)),inp)) for inp in split_inputs]
        # print(len(split_inputs))
        # print(split_inputs[0].shape)
        # sadfsdf
        # print(split_token_type_ids[0])
        # print(split_inputs[0].shape)
        # kzjfhsd
        for inp_id, inp in enumerate(split_inputs):
        	new_input_ids = inp.tolist()
        	new_token_type_ids = torch.cat((torch.tensor([0]*len(genre_tags)), split_token_type_ids[inp_id])).tolist()
        	attention_mask = [1 if mask_padding_with_zero else 0] * len(new_input_ids)
        	padding_length = max_length - len(new_input_ids)
        	if padding_length > 0:
        		# print(new_input_ids.data)
        		new_input_ids = new_input_ids + ([pad_token] * padding_length)
        		attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        		new_token_type_ids = new_token_type_ids + ([pad_token_segment_id] * padding_length)
        	# print(new_input_ids)
        	# print(new_token_type_ids)
        	# print(attention_mask)
        	# print(pad_token_segment_id) 
        	assert len(new_input_ids) == max_length
        	assert len(attention_mask) == max_length
        	assert len(new_token_type_ids) == max_length
        	choices_features.append((new_input_ids, attention_mask, new_token_type_ids))
        	features.append(InputFeatures(example_id=example.title, choices_features=choices_features))
        max_tokens = outer_max_tokens 
        # if count == 500:
        # 	return features
        count+=1
    return features

