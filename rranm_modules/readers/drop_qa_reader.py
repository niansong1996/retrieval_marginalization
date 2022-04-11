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

@DatasetReader.register('drop-qa-reader')
class DropQAReader(IIRCQAReader):
    def __init__(self, wiki_file_path, allow_skipping, **kwargs) -> None:
        super().__init__(wiki_file_path, no_loading_wiki_dict=True, **kwargs)
        self.allow_skipping = allow_skipping

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info("Reading dataset file at %s", file_path)
        with open(file_path, 'r') as f:
            dataset = json.load(f)

            for _, init_context_json in dataset.items():
                context_text = init_context_json["passage"]

                for qa_json in init_context_json["qa_pairs"]:
                    question_text = qa_json["question"]
                    answer_dict = qa_json["answer"]

                    qa_fields = self.get_qa_fields(question_text, context_text, answer_dict,
                                                   allow_skipping=self.allow_skipping)

                    if qa_fields is None:
                        logger.info('skipping example for invalid gold context')
                        continue
                    else:
                        yield Instance(qa_fields)
