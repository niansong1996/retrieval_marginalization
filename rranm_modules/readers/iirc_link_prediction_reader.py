import json
import pathlib
import re
import logging
import sys
import numpy as np

from overrides import overrides
from typing import Dict, Iterable, List, Tuple
from functools import reduce

from allennlp.data import DatasetReader, Instance
from .iirc_reader import TransformerIIRCReader
from allennlp.data.fields import Field, TextField, IndexField, ListField, SpanField, MetadataField, ArrayField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer

logger = logging.getLogger('reader')


@DatasetReader.register('iirc-link-prediction-reader')
class IIRCLinkPredictionReader(TransformerIIRCReader):
    def __init__(self, wiki_file_path: str, use_answer_links: bool = False, **kwargs):
        super().__init__(wiki_file_path, **kwargs)
        self.use_answer_links = use_answer_links

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info("Reading dataset file at %s", file_path)
        with open(file_path, 'r') as f:
            dataset = json.load(f)

            for context_questions in dataset:
                init_context = context_questions['text']
                all_links = list(map(lambda x: (x['indices'], x['target'].lower()), context_questions['links']))

                for question_json in context_questions['questions']:
                    fields = self.get_link_prediction_fields(init_context, all_links,
                                                             question_json, self.use_answer_links)
                    yield Instance(fields)

