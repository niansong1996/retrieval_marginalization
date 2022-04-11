import re
import json
import string
import itertools
import numpy as np
import torch
import logging

from overrides import overrides
from typing import List, Dict, Any, Tuple, Union, Iterable
from tqdm import tqdm
from word2number.w2n import word_to_num
from collections import defaultdict
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, TextField, IndexField, ListField, SpanField, MetadataField, ArrayField
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer

from .iirc_qa_reader import IIRCQAReader

logger = logging.getLogger('reader')

@DatasetReader.register('hotpot-qa-reader')
class HotpotQAReader(IIRCQAReader):
    def __init__(self, wiki_file_path, allow_skipping, **kwargs) -> None:
        super().__init__(wiki_file_path, no_loading_wiki_dict=True, **kwargs)

        self.allow_skipping = allow_skipping

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info("Reading dataset file at %s", file_path)
        with open(file_path, 'r') as f:
            dataset = json.load(f)

            for example in dataset:
                question_text = example["question"]
                try:
                    float(example["answer"])
                    answer_dict = {'date': {'day': '', 'month': '', 'year': ''},
                                   'number': example["answer"], 'spans': []}
                except ValueError:
                    answer_dict = {'date': {'day': '', 'month': '', 'year': ''},
                                   'number': '', 'spans': [example["answer"]]}

                # first formulate the dictionary to map title to sentences
                title_sents_dict = dict(map(lambda x: (x[0], x[1]), example["context"]))

                context_sents_list = []
                for title, sent_idx in example["supporting_facts"]:
                    if sent_idx < len(title_sents_dict[title]):
                        context_sents_list.append(title_sents_dict[title][sent_idx])
                    else:
                        context_sents_list = []
                        break

                if len(context_sents_list) == 0:
                    logger.info('skipping example for incorrect idx or no context')
                    continue

                context_text = (" " + self._tokenizer_sep_token.text + " ").join(context_sents_list)

                qa_fields = self.get_qa_fields(question_text, context_text, answer_dict,
                                               allow_skipping=self.allow_skipping)

                if qa_fields is None:
                    logger.info('skipping example for invalid gold context')
                    continue
                else:
                    yield Instance(qa_fields)
