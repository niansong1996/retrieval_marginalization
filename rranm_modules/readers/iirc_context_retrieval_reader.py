import json
import logging
import spacy
import numpy as np

from overrides import overrides
from typing import Dict, Iterable, List, Tuple
from allennlp.data.fields import Field, TextField, IndexField, ListField, SpanField, MetadataField, ArrayField

from allennlp.data import DatasetReader, Instance
from .iirc_reader import TransformerIIRCReader

logger = logging.getLogger('reader')


@DatasetReader.register('iirc-context-retrieval-reader')
class IIRCContextRetrievalReader(TransformerIIRCReader):
    def __init__(self, wiki_file_path: str, cache_directory: str = None, sent_n: int = 1, padding_sent_n: int = 1,
                 stride: int = 1, neg_n: int = 7, include_main: bool = False, max_neg_n: int = 500,
                 add_ctx_sep: bool = False, link_per_question: int = 3, add_init_context: bool = False, **kwargs):
        super().__init__(wiki_file_path, cache_directory, **kwargs)
        self.sent_n = sent_n
        self.padding_sent_n = padding_sent_n
        self.stride = stride
        self.neg_n = neg_n
        self.include_main = include_main
        self.max_neg_n = max_neg_n
        self.add_ctx_sep = add_ctx_sep
        self.link_per_question = link_per_question
        self.add_init_context = add_init_context

        self.spacy_pipeline = spacy.load("en", disable=["tagger", "parser", "ner"])
        self.spacy_pipeline.add_pipe(self.spacy_pipeline.create_pipe("sentencizer"))

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info("Reading dataset file at %s", file_path)
        with open(file_path, 'r') as f:
            dataset = json.load(f)

            for context_questions in dataset:
                init_context = context_questions['text']
                init_context_sents = self.spacy_pipeline(init_context).sents
                all_link_name_pos_tuple = list(map(lambda x: (x['target'].lower(), x['indices'][0]),
                                                   context_questions['links']))
                link_name_init_sent_dict = dict()

                for sent in init_context_sents:
                    s, e = sent.start_char, sent.end_char
                    for name, pos in all_link_name_pos_tuple:
                        if s <= pos < e and name not in link_name_init_sent_dict:
                            link_name_init_sent_dict[name] = sent.string

                # every question constitutes an instance
                for question_json in context_questions['questions']:
                    context_title_text_dict = dict()

                    # get all the gold links and fill the rest with randomly select irrelevant link
                    gold_links_dict: Dict[str, Dict] = dict()
                    for context_json in question_json['context']:
                        # first check if the context is the initiating context
                        context_title = context_json['passage'].lower()
                        if context_title != 'main' or self.include_main:
                            gold_links_dict[context_title] = context_json
                            context_title_text_dict[context_title] = context_json['text']

                    all_non_gold_links_set = set(filter(lambda y: y not in gold_links_dict and y in self.wiki_dict,
                                                 map(lambda x: x['target'].lower(), context_questions['links'])))
                    all_non_gold_links = list(map(lambda x: (x, None), all_non_gold_links_set))
                    np.random.shuffle(all_non_gold_links)
                    all_link_candidates = (list(gold_links_dict.items()) + all_non_gold_links)[:self.link_per_question]
                    np.random.shuffle(all_link_candidates)
                    if not len(all_link_candidates) == self.link_per_question:
                        logging.info('skipping invalid examples of size ' + str(len(all_link_candidates)))
                        continue

                    # get the fields from every evidence json into a list
                    context_sents, sent_indices, correct_context_mask = [], [], []
                    link_name_list, raw_sents_list = [], []
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
                        link_name_list.append(context_title)
                        raw_sents_list.append(context_fields['raw_sents'])

                    if invalid:
                        logging.info('skipping invalid examples...')
                        continue

                    fields = dict()
                    fields['context_sents'] = ListField(context_sents)
                    fields['sent_indices'] = ListField(sent_indices)
                    fields['correct_context_mask'] = ListField(correct_context_mask)
                    fields['link_name_list'] = MetadataField(link_name_list)
                    fields['question_text'] = MetadataField(question_json['question'])
                    fields['raw_sents_list'] = MetadataField(raw_sents_list)
                    fields['context_title_text_dict'] = MetadataField(context_title_text_dict)

                    yield Instance(fields)
