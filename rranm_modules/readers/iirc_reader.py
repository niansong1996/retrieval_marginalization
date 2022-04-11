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
from allennlp.data.fields import Field, TextField, IndexField, ListField, SpanField, MetadataField, ArrayField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer

from rranm_modules.utils.tokenizer_utils import add_special_tokens_to_sent_pair


logger = logging.getLogger('reader')


class TransformerIIRCReader(DatasetReader):
    def __init__(self,
                 wiki_file_path: str,
                 cache_directory: str = None,
                 transformer_model_name: str = "bert-base-uncased",
                 q_max_tokens: int = 64,
                 c_max_tokens: int = 384,
                 skip_invalid_examples: bool = False,
                 no_loading_wiki_dict: bool = False,
                 **kwargs):
        super().__init__(cache_directory=cache_directory, **kwargs)
        self._tokenizer = PretrainedTransformerTokenizer(transformer_model_name, add_special_tokens=False)
        self._tokenizer_sep_token = self._tokenizer.sequence_pair_mid_tokens[0]
        self._tokenizer_start_token = self._tokenizer.sequence_pair_start_tokens[0]
        self._tokenizer_end_token = self._tokenizer.sequence_pair_end_tokens[0]
        self._token_indexers = {"tokens": PretrainedTransformerIndexer(transformer_model_name)}

        # set the model constants
        self.q_max_tokens = q_max_tokens if q_max_tokens else sys.maxsize
        self.c_max_tokens = c_max_tokens if c_max_tokens else sys.maxsize
        self.skip_invalid_examples = skip_invalid_examples

        # read wiki documents
        if not no_loading_wiki_dict:
            logger.info('start reading wikipedia documents...')
            with open(wiki_file_path, 'r') as f:
                self.wiki_dict = dict(json.load(f, strict=False))
                self.wiki_dict['null context'] = [{'text': 'NULL', 'start_idx': 0, 'end_idx': 3}]
            logger.info('{} wikipedia documents are loaded'.format(len(self.wiki_dict)))
        else:
            logger.info('skipping loading wikipedia documents')


    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        raise ValueError('The TransformerIIRCReader class is an abstract class and shouldn\'t be initialized!')

    @overrides
    def text_to_instance(self, init_context: str,
                         all_links: List[Tuple[List[int], str]],
                         title: str,
                         question_json: dict) -> Instance:
        raise ValueError('text_to_instance(...) method is not used for iirc reader!')

    def get_link_prediction_fields(self, init_context: str,
                                   all_links: List[Tuple[List[int], str]],
                                   question_json: dict, use_answer_links: bool) -> (Dict[str, Field], Dict):
        """
        Get the fields for link prediction, operates on question-level:
            init_context -> List[questions] <- question

        Returns:
        -------
        A dictionary of the following fields:
            'question_with_context': `TextField` of tokenized concatenated question and init_context \n
            'all_links_mask': `ArrayField` of a mask of all the positions with links being 1, others being 0 \n
            'gold_links_mask': `ArrayField` of a mask of all the gold link positions being 1, others being 0 \n
        and a dictionary of:
            k: link names \n
            v: link start token position \n
        """

        # first get all the raw information from question_json
        question = question_json['question']
        all_links_set_tmp = set(map(lambda x: x[1].lower(), all_links))
        if use_answer_links:
            gold_links = set(filter(lambda y: y != 'main',
                                     map(lambda x: x['passage'].lower(), question_json['context'])))
        else:
            gold_links = set(filter(lambda y: y in self.wiki_dict and y in all_links_set_tmp,
                                    map(lambda x: x.lower(), question_json['question_links'])))

        # fill in all the fields
        fields = dict()

        # Tokenize the question and the context
        tokenized_question = self._tokenizer.tokenize(question)[: self.q_max_tokens]
        tokenized_context = self._tokenizer.tokenize(init_context)[:self.c_max_tokens]
        tokenized_qc = add_special_tokens_to_sent_pair(self._tokenizer, tokenized_question, tokenized_context)

        # find the token index of the link starting words
        all_links = sorted(all_links, key=lambda x: x[0][0])
        link_indices = []
        i = len(tokenized_question) + 1
        for link in all_links:
            while i < len(tokenized_qc) and (tokenized_qc[i].idx is None or tokenized_qc[i].idx < link[0][0]):
                i += 1
            if i < len(tokenized_qc):
                link_indices.append((link[1], i))
        all_links_indices = list(map(lambda x: x[1], link_indices))
        all_links_mask = np.zeros_like(tokenized_qc, dtype=float)
        all_links_mask[all_links_indices] = 1.0

        gold_links_indices = list(map(lambda x: x[1], filter(lambda x: x[0] in gold_links, link_indices)))
        gold_links_mask = np.zeros_like(tokenized_qc, dtype=float)
        gold_links_mask[gold_links_indices] = 1.0

        link_token_pos_name_dict = dict(map(lambda x: (x[1], x[0]), link_indices))

        fields['question_with_context'] = TextField(tokenized_qc, self._token_indexers)
        fields['all_links_mask'] = ArrayField(all_links_mask)
        fields['gold_links_mask'] = ArrayField(gold_links_mask)
        fields['question_text'] = MetadataField(question)
        fields['link_pos_name_dict'] = MetadataField(link_token_pos_name_dict)
        fields['gold_link_names'] = MetadataField(gold_links)

        return fields

    def get_context_retrieval_fields(self, question_text: str, document_sents: List[str], context_json: dict,
                                     sent_n: int, padding_sent_n: int, stride: int,
                                     neg_n: int, max_neg_n: int, add_ctx_sep: bool,
                                     allow_all_neg: bool, all_neg_indicator: bool,
                                     add_init_context: bool, init_context_text: str) -> Dict[str, Field]:
        """
        Get the fields for context retrieval, operates on evidence-level:
            init_context -> List[questions] -> List[List[evidence]] <- evidence

        Returns:
        -------
        A dictionary of the following fields:
            'context_sents': `ListField[TextField]` a list of possible chunks of candidates containing the evidence \n
            'sent_indices': `ListField[IndexField]` a list of sent start positions of the chunks mentioned above \n
            'correct_context_mask': `ArrayField` a mask over all the possible chunks w/ correct chunk 1, others 0 \n
        """
        # preprocess the document sentences to remove blanks
        document_sents = list(filter(lambda x: len(x) > 0, map(lambda y: y.strip(), document_sents)))

        # if this document contains supporting facts (context_json not None) get the index of the sentence that it is in
        if context_json is not None:
            context_sent_idx, _ = context_json['sent_indices']
        else:
            context_sent_idx = -1

        # make chunks of the document text using sliding window
        negative_examples: List[Tuple[List[str], str, int, int]] = []
        positive_examples: List[Tuple[List[str], str, int, int]] = []
        for i in range(0, len(document_sents), stride):
            sent = document_sents[i]
            sent_with_context = document_sents[max(i-padding_sent_n, 0):i+sent_n+padding_sent_n]
            sent_idx = min(i, padding_sent_n)

            if i == context_sent_idx:
                positive_examples.append((sent_with_context, sent, sent_idx, 1))
            else:
                negative_examples.append((sent_with_context, sent, sent_idx, 0))

        # no instance should be generated when the doc does not contain evidence and all neg is not allowed
        positive_example_n = len(positive_examples)
        if positive_example_n == 0 and not allow_all_neg:
            logging.info('no positive example found and all negative is not allowed, skip the instance')
            return {}
        if len(negative_examples) == 0:
            logging.info('no negative example for the document, skip the instance')
            return {}

        # down-sampling the negative examples and mix with the positive one
        if neg_n != -1:
            neg_sample_size = neg_n+1-positive_example_n
            negative_examples = [negative_examples[i]
                                 for i in np.random.choice(len(negative_examples), neg_sample_size).tolist()]
        else:
            negative_examples = negative_examples[:max_neg_n]
        examples: List[Tuple[List[str], str, int, int]] = negative_examples + positive_examples
        np.random.shuffle(examples)

        # tokenize the sents, max token cutoff and add special tokens
        results: List[Tuple[TextField, IndexField, str, int]] = []
        for sent_with_context, raw_sent, sent_idx, is_positive in examples:
            # tokenize the chunk of sentences
            main_token_list, left_pad_token_list, right_pad_token_list = [], [], []
            for j in range(len(sent_with_context)):
                tokens = self._tokenizer.tokenize(sent_with_context[j])
                if j < sent_idx:
                    left_pad_token_list += tokens
                elif sent_idx <= j < sent_idx+sent_n:
                    main_token_list += tokens
                else:
                    right_pad_token_list += tokens
            lc, mc, rc = len(left_pad_token_list), len(main_token_list), len(right_pad_token_list)

            # chop off from end first then from start
            if lc + mc + rc > self.c_max_tokens:
                # see if we can keep the whole left padding
                if lc + mc < self.c_max_tokens:
                    right_pad_token_list = right_pad_token_list[:self.c_max_tokens - (lc+mc)]
                    sent_start_token_idx = lc
                # if not, see if we can keep part of the left padding
                elif mc < self.c_max_tokens:
                    right_pad_token_list = []
                    left_space = self.c_max_tokens - mc
                    left_pad_token_list = left_pad_token_list[-left_space:]
                    sent_start_token_idx = left_space
                # now we even need to cut off the main sent tokens
                else:
                    left_pad_token_list = []
                    right_pad_token_list = []
                    main_token_list = main_token_list[:self.c_max_tokens]
                    sent_start_token_idx = 0
            else:
                sent_start_token_idx = lc

            # now we add some special tokens (or not)
            if add_ctx_sep:
                token_list = left_pad_token_list + self._tokenizer.sequence_pair_mid_tokens \
                             + main_token_list + self._tokenizer.sequence_pair_mid_tokens \
                             + right_pad_token_list
            else:
                token_list = left_pad_token_list + main_token_list + right_pad_token_list

            # detect if there are bad examples and deal with them accordingly
            try:
                assert 0 <= sent_start_token_idx < len(token_list)
            except AssertionError:
                if is_positive and not allow_all_neg:
                    logging.info('get bad positive example, skip the instance')
                    return {}
                else:
                    logging.info('get bad negative example, substitute with new sampled example')
                    random_idx = np.random.choice(len(negative_examples), 1).tolist()[0]
                    examples.append(negative_examples[random_idx])
                    continue

            # init the two fields
            question_tokens = self._tokenizer.tokenize(question_text)[:self.q_max_tokens]
            if add_init_context:
                init_context_tokens = self._tokenizer.tokenize(init_context_text)
                token_list = (token_list + self._tokenizer.sequence_pair_mid_tokens + init_context_tokens)[:self.c_max_tokens]
            qc_tokens = add_special_tokens_to_sent_pair(self._tokenizer, question_tokens, token_list)
            text_field = TextField(qc_tokens, self._token_indexers)
            sent_start_token_idx += len(question_tokens) + 2
            index_field = IndexField(sent_start_token_idx, text_field)

            results.append((text_field, index_field, raw_sent, is_positive))

        # add the all negative dummy chunk as the all negative indicator
        if all_neg_indicator:
            question_tokens = self._tokenizer.tokenize(question_text)[:self.q_max_tokens]
            if add_init_context:
                init_context_tokens = self._tokenizer.tokenize(init_context_text)
                token_list = (self._tokenizer.tokenize('NULL') + self._tokenizer.sequence_pair_mid_tokens
                              + init_context_tokens)[:self.c_max_tokens]
            qc_tokens = add_special_tokens_to_sent_pair(self._tokenizer, question_tokens,
                                                        token_list)
            text_field = TextField(qc_tokens, self._token_indexers)
            sent_start_token_idx = len(question_tokens) + 2
            index_field = IndexField(sent_start_token_idx, text_field)
            is_positive = 0 if sum(map(lambda x: x[3], results)) > 0 else 1

            results.append((text_field, index_field, 'NULL', is_positive))

        # return the fields dict
        fields = dict()

        fields['context_sents'] = ListField(list(map(lambda x: x[0], results)))
        fields['sent_indices'] = ListField(list(map(lambda x: x[1], results)))
        fields['raw_sents'] = MetadataField(list(map(lambda x: x[2], results)))
        fields['correct_context_mask'] = ArrayField(np.array(list(map(lambda x: x[3], results))))
        # fields['positive_n'] = ArrayField(np.array([positive_example_n]))

        return fields

    def get_metadata_fields(self, question_text: str, context_json_list: List[dict]) -> Dict[str, Field]:
        """
        Get the raw metadata for debugging and interpretation, operates on question-level:
            init_context -> List[questions] <- question

        Returns:
        -------
        A dictionary of the following fields:
            'question_text': `MetadataField[str]` the raw question text \n
            'context_title_text_dict': `MetadataField[Dict[str, str]]` a dictionary mapping evidence title to evidence
                spans \n
        """

        fields: Dict[str, Field] = dict()
        fields['question_text'] = MetadataField(question_text)

        # add the field for all supporting evidences
        context_dict = dict()
        for context_json in context_json_list:
            if context_json is not None:
                context_dict[context_json['passage'].lower()] = context_json['text']
        fields['context_title_text_dict'] = MetadataField(context_dict)

        return fields








if __name__ == '__main__':
    # reader = IIRCContextRetrievalReader('../../data/iirc/context_articles.json')
    # reader.read('../../data/iirc/iirc_tiny.json')


    with open('../../data/iirc/preprocessed_iirc_train.json') as f:
        dataset = json.load(f)

        context_n = 0
        context_first_gold_n = 0
        for context_questions in dataset:
            init_context = context_questions['text']
            all_links = list(map(lambda x: (x['indices'], x['target']), context_questions['links']))
            title = context_questions['title']

            for question_json in context_questions['questions']:
                for context_dict in question_json['context']:
                    context_n += 1
                    if context_dict['sent_indices'][0] == 0:
                        context_first_gold_n += 1


    with open('../../data/iirc/context_articles.json', 'r') as f:
        wiki_dict = json.load(f)

    print("")