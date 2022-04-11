import json
import pathlib
import re
import logging
import spacy
import sys
import numpy as np

from overrides import overrides
from typing import Dict, Iterable, List, Tuple
from functools import reduce

from allennlp.data import DatasetReader, Instance
from .iirc_joint_retrieval_reader import IIRCJointRetrievalReader

logger = logging.getLogger('reader')


@DatasetReader.register('iirc-joint-qa-reader')
class IIRCJointQAReader(IIRCJointRetrievalReader):
    def __init__(self, wiki_file_path,
                 sent_n: int = 1, padding_sent_n: int = 1, stride: int = 1,
                 neg_n: int = 7, include_main: bool = False, max_neg_n: int = 500,
                 use_answer_links: bool = False, add_ctx_sep: bool = False,
                 add_init_context: bool = False,
                 link_per_question: int = 3, **kwargs) -> None:
        super().__init__(wiki_file_path, sent_n, padding_sent_n, stride, neg_n, include_main, max_neg_n,
                         use_answer_links, add_ctx_sep, add_init_context, link_per_question, **kwargs)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        return self._read_retrieval_fields_with_answer_option(file_path,
                                                              include_answer_dict=True,
                                                              include_original_paragraph=True)
