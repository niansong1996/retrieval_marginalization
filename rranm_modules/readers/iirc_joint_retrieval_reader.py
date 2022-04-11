import json
import pathlib
import re
import logging
import sys
import numpy as np
import spacy

from overrides import overrides
from typing import Dict, Iterable, List, Tuple
from functools import reduce

from allennlp.data import DatasetReader, Instance
from .iirc_reader import TransformerIIRCReader
from .iirc_qa_reader import iirc_answer_to_drop_style
from allennlp.data.fields import Field, TextField, IndexField, ListField, SpanField, MetadataField, ArrayField

logger = logging.getLogger('reader')


@DatasetReader.register('iirc-joint-retrieval-reader')
class IIRCJointRetrievalReader(TransformerIIRCReader):
    def __init__(self, wiki_file_path: str, sent_n: int = 1, padding_sent_n: int = 1, stride: int = 1,
                 neg_n: int = 7, include_main: bool = False, max_neg_n: int = 500,
                 use_answer_links: bool = False, add_ctx_sep: bool = False,
                 add_init_context: bool = False,
                 link_per_question: int = 3, **kwargs):
        super().__init__(wiki_file_path, **kwargs)
        self.sent_n = sent_n
        self.padding_sent_n = padding_sent_n
        self.stride = stride
        self.neg_n = neg_n
        self.include_main = include_main
        self.max_neg_n = max_neg_n
        self.use_answer_links = use_answer_links
        self.add_ctx_sep = add_ctx_sep
        self.add_init_context = add_init_context
        self.link_per_question = link_per_question

        self.spacy_pipeline = spacy.load("en", disable=["tagger", "parser", "ner"])
        self.spacy_pipeline.add_pipe(self.spacy_pipeline.create_pipe("sentencizer"))

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        return self._read_retrieval_fields_with_answer_option(file_path,
                                                              include_answer_dict=False,
                                                              include_original_paragraph=False)

    def _read_retrieval_fields_with_answer_option(self, file_path: str,
                                                  include_answer_dict: bool,
                                                  include_original_paragraph: bool)\
            -> Iterable[Instance]:
        logger.info("Reading dataset file at %s", file_path)
        with open(file_path, 'r') as f:
            dataset = json.load(f)

            for context_questions in dataset:
                init_context = context_questions['text']

                # insert a dummy link; see which part is not covered by existing link
                cover_list = np.zeros(len(init_context))
                for x in context_questions['links']:
                    cover_list[x['indices'][0]:(x['indices'][1]+1)] = 1

                dummy_link_pos = list(filter(lambda x: x[0] > 5
                                                       and x[0] + 5 < len(cover_list)
                                                       and cover_list[x[0]-2] ==0
                                                       and cover_list[x[0]-1] ==0
                                                       and cover_list[x[0]] ==0
                                                       and cover_list[x[0]+1] ==0
                                                       and cover_list[x[0]+2] ==0
                                                       and x[1] == 0,
                                             list(enumerate(cover_list))))[0][0]
                context_questions['links'] += [{'indices': [dummy_link_pos, dummy_link_pos + 1],
                                                'target': 'null context'}]

                init_context_sents = self.spacy_pipeline(init_context).sents
                all_links_indices_name_tuple = list(filter(lambda y: y[1] in self.wiki_dict,
                                                    map(lambda x: (x['indices'], x['target'].lower()),
                                                        context_questions['links'])))
                link_name_init_sent_dict = dict()

                for sent in init_context_sents:
                    s, e = sent.start_char, sent.end_char
                    for indices, name in all_links_indices_name_tuple:
                        if s <= indices[0] < e and name not in link_name_init_sent_dict:
                            link_name_init_sent_dict[name] = sent.string

                # every question constitutes an instance
                for question_json in context_questions['questions']:
                    context_title_text_dict = dict()

                    # get the link prediction fields
                    link_prediction_fields = self.get_link_prediction_fields(init_context, all_links_indices_name_tuple,
                                                                             question_json, self.use_answer_links)

                    # get all the gold links and fill the rest with randomly select irrelevant link
                    gold_links_dict: Dict[str, Dict] = dict()
                    for context_json in question_json['context']:
                        # first check if the context is the initiating context
                        context_title = context_json['passage'].lower()
                        if context_title != 'main' or self.include_main:
                            gold_links_dict[context_title] = context_json
                            context_title_text_dict[context_title] = context_json['text']

                    assert 'null context' in self.wiki_dict
                    assert 'null context' in map(lambda x: x['target'].lower(), context_questions['links']), str(context_questions['links'])
                    assert 'null context' not in gold_links_dict
                    all_non_gold_links_set = set(filter(lambda y: y not in gold_links_dict and y in self.wiki_dict,
                                                        map(lambda x: x['target'].lower(), context_questions['links'])))
                    all_non_gold_links = list(map(lambda x: (x, None), all_non_gold_links_set))
                    np.random.shuffle(all_non_gold_links)
                    all_link_candidates = (list(gold_links_dict.items()) + all_non_gold_links)
                    np.random.shuffle(all_link_candidates)

                    if len(all_link_candidates) < self.link_per_question:
                        logging.info('skipping invalid examples of size ' + str(len(all_link_candidates))
                                     + ' and link_per_question is ' + str(self.link_per_question))
                        continue

                    # a mapping from link name to the idx in the all links list
                    link_name_idx_dict = dict(map(lambda i, x: (x[0], i),
                                                  range(len(all_link_candidates)), all_link_candidates))

                    # get the fields from every evidence json into a list
                    context_sents, sent_indices, correct_context_mask = [], [], []
                    raw_sents_list = []
                    invalid = False
                    for context_title, context_json in all_link_candidates:
                        # get the document raw text and get the fields for creating an instance
                        question_text = question_json['question']
                        document_text = self.wiki_dict[context_title]
                        document_text_sents = list(map(lambda x: x['text'], document_text))
                        init_context_text = link_name_init_sent_dict[context_title]
                        context_fields = self.get_context_retrieval_fields(question_text, document_text_sents,
                                                                           context_json, self.sent_n,
                                                                           self.padding_sent_n, self.stride,
                                                                           self.neg_n, self.max_neg_n, self.add_ctx_sep,
                                                                           allow_all_neg=True, all_neg_indicator=True,
                                                                           add_init_context=self.add_init_context,
                                                                           init_context_text=init_context_text)

                        if len(context_fields) == 0:
                            invalid = True
                            break

                        # add the individual context fields to the question
                        context_sents.append(context_fields['context_sents'])
                        sent_indices.append(context_fields['sent_indices'])
                        correct_context_mask.append(context_fields['correct_context_mask'])
                        raw_sents_list.append(context_fields['raw_sents'])

                    if invalid:
                        logging.info('skipping invalid examples...')
                        continue

                    fields = dict()
                    fields['context_sents_list'] = ListField(context_sents)
                    fields['sent_indices_list'] = ListField(sent_indices)
                    fields['correct_context_mask_list'] = ListField(correct_context_mask)
                    if 'null context' not in self.wiki_dict:
                        print('not in wiki dict')
                    fields['link_name_idx_dict'] = MetadataField(link_name_idx_dict)
                    fields['raw_sents_list'] = MetadataField(raw_sents_list)
                    fields['context_title_text_dict'] = MetadataField(context_title_text_dict)

                    # those fields include: 'question_with_context', 'all_links_mask', 'gold_links_mask'
                    # 'question_text', 'link_pos_name_dict', 'gold_link_names'
                    fields.update(link_prediction_fields)

                    # include the answer dict if the option is set to true
                    if include_answer_dict:
                        answer_dict = iirc_answer_to_drop_style(question_json["answer"])
                        fields['answer_dict'] = MetadataField(answer_dict)

                        if answer_dict is None:
                            logger.info(f'skipping instance for lack of well-formatted answer...')
                            continue

                    if include_original_paragraph:
                        fields['original_paragraph'] = MetadataField(init_context)

                    yield Instance(fields)
