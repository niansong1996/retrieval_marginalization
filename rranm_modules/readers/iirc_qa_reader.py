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

from .iirc_reader import TransformerIIRCReader

IGNORED_TOKENS = {'a', 'an', 'the'}
STRIPPED_CHARACTERS = string.punctuation + ''.join([u"‘", u"’", u"´", u"`", "_"])
USTRIPPED_CHARACTERS = ''.join([u"Ġ"])

logger = logging.getLogger('reader')


@DatasetReader.register('iirc-qa-reader')
class IIRCQAReader(TransformerIIRCReader):
    def __init__(self, wiki_file_path,
                 skip_when_all_empty: List[str] = None,
                 relaxed_span_match_for_finding_labels: bool = True,
                 use_retrieved_context: bool = False,
                 retrieved_context_file: str = '', **kwargs) -> None:
        super().__init__(wiki_file_path, **kwargs)

        self.skip_when_all_empty = skip_when_all_empty if skip_when_all_empty is not None else []
        self.use_retrieved_context = use_retrieved_context
        self.retrieved_context_file = retrieved_context_file

        # jferguson Done
        for item in self.skip_when_all_empty:
            assert item in ["passage_span", "question_span", "addition_subtraction",
                            "counting", "none", "binary"], "Unsupported skip type: %s" % (item)
        self.relaxed_span_match_for_finding_labels = relaxed_span_match_for_finding_labels

    def load_context(self):
        with open(self.retrieved_context_file, 'r') as f:
            lines = f.readlines()

        context_dict = dict()
        for line in lines:
            if '@@@' not in line:
                print(line)
                continue
            question = line.split('@@@')[0].lower().strip()
            contexts = [a.strip() for a in line.split('@@@')[1].split('###')]
            contexts = list(filter(lambda x: x != 'NULL', contexts))

            context_dict[question] = contexts

        return context_dict

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info("Reading dataset file at %s", file_path)
        if self.use_retrieved_context:
            context_dict = self.load_context()
        else:
            context_dict = None

        with open(file_path, 'r') as f:
            # all_instances: List[Instance] = []
            total_n, skipped_n = 0, 0
            dataset = json.load(f)
            for passage in dataset:
                original_text = passage["text"]
                for question in passage["questions"]:
                    question_text = question["question"]

                    if self.use_retrieved_context:
                        if question_text.lower().strip() not in context_dict:
                            logger.info(f'question {question_text} is not in the retrieved output question list')

                        # text_segments = [original_text]
                        text_segments = []
                        text_segments += context_dict.get(question_text.lower().strip(), [])
                        text_segments.append(original_text)
                        # text_segments.append("PAD")  # to bypass some edge cases
                        new_text = (" " + self._tokenizer_sep_token.text + " ").join(text_segments)

                    else:
                        if not question["context"]:
                            new_text = original_text
                        else:
                            text_segments = []
                            # text_segments = [original_text]
                            for context_span in question["context"]:
                                context_title = context_span["passage"].lower()
                                context_start, context_end = context_span["indices"]
                                if context_title == "main":
                                    # text_segments.append(original_text[context_start:context_end])
                                    continue
                                elif context_title in self.wiki_dict:
                                    text_segments.append(span_sentences(self.wiki_dict[context_title],
                                                                        context_start, context_end))
                                else:
                                    continue

                            # new_text = self._tokenizer_start_token.text + " " + \
                            #            (" " + self._tokenizer_sep_token.text + " ").join(text_segments) \
                            #            + " " + self._tokenizer_end_token.text
                            text_segments.append(original_text)

                            new_text = (" " + self._tokenizer_sep_token.text + " ").join(text_segments)

                    answer_dict = iirc_answer_to_drop_style(question["answer"])

                    if answer_dict is None:
                        continue

                    # question_text = self._tokenizer_start_token.text \
                    #                 + " " + question_text + " " + self._tokenizer_end_token.text

                    fields = self.get_qa_fields(question_text, new_text, answer_dict)
                    if fields is not None:
                        total_n += 1
                        # all_instances.append(Instance(fields))
                        yield Instance(fields)
                    else:
                        skipped_n += 1

            # logger.info(f'{total_n} valid instance with {skipped_n} skipped')
            # return all_instances

    def get_qa_fields(self, question_text: str, context_text: str,
                      answer_dict: Dict, allow_skipping: bool = True) -> Union[Dict[str, Field], None]:
        # context_text = " ".join(whitespace_tokenize(context_text))
        # question_text = " ".join(whitespace_tokenize(question_text))

        # get the numbers in the context and question
        passage_tokens, passage_offset, numbers_in_passage, passage_number_indices, number_len = \
            roberta_tokenize(context_text, self._tokenizer)
        question_tokens, question_offset, numbers_in_question, question_number_indices, question_number_len = \
            roberta_tokenize(question_text, self._tokenizer)

        # clip the question and passage along with the numbers and their indices
        if self.c_max_tokens is not None:
            passage_tokens = passage_tokens[: self.c_max_tokens]
            if len(passage_number_indices) > 0:
                passage_number_indices, number_len, numbers_in_passage = \
                    clipped_passage_num(passage_number_indices, number_len, numbers_in_passage, len(passage_tokens))

        if self.q_max_tokens is not None:
            question_tokens = question_tokens[: self.q_max_tokens]
            if len(question_number_indices) > 0:
                question_number_indices, question_number_len, numbers_in_question = \
                    clipped_passage_num(question_number_indices, question_number_len, numbers_in_question,
                                        len(question_tokens))

        # get the answer type(str) and texts (List[str])
        answer_type, answer_texts = self.extract_answer_info_from_annotation(answer_dict)

        # Tokenize the answer text in order to find the matched span based on token
        tokenized_answer_texts: List[str] = []

        for answer_text in answer_texts:
            answer_tokens, _, _, _, _ = roberta_tokenize(answer_text, self._tokenizer, True)
            tokenized_answer_texts.append(' '.join([token.text for token in answer_tokens]))

        all_number = numbers_in_passage + numbers_in_question
        all_number_order = get_number_order(all_number)

        # TODO: highly doubt whether the next 3 lines are useful
        if len(all_number_order) == 0:
            passage_number_order = []
            question_number_order = []
        else:
            passage_number_order = all_number_order[:len(numbers_in_passage)]
            question_number_order = all_number_order[len(numbers_in_passage):]

        # TODO: is this to account for the sentence start token? may not be necessary since we already appended it
        passage_number_indices = [indice + 1 for indice in passage_number_indices]
        numbers_in_passage.append(100)
        passage_number_indices.append(0)
        passage_number_order.append(-1)

        # hack to guarantee minimal length of padded number
        numbers_in_passage.append(0)
        passage_number_indices.append(-1)
        passage_number_order.append(-1)

        numbers_in_question.append(0)
        question_number_indices.append(-1)
        question_number_order.append(-1)

        passage_number_order = np.array(passage_number_order)
        question_number_order = np.array(question_number_order)

        numbers_as_tokens: List[str] = [str(number) for number in numbers_in_passage]

        # if we can find the span in the question, then we don't try to find it in the question TODO: but why
        valid_passage_spans = self.find_valid_spans(passage_tokens, tokenized_answer_texts)
        if len(valid_passage_spans) > 0:
            valid_question_spans = []
        else:
            valid_question_spans = self.find_valid_spans(question_tokens, tokenized_answer_texts)

        target_numbers = []
        # `answer_texts` is a list of valid answers.
        for answer_text in answer_texts:
            number = get_number_from_word(answer_text, True)
            if number is not None:
                target_numbers.append(number)
        valid_signs_for_add_sub_expressions: List[List[int]] = []
        valid_counts: List[int] = []
        if answer_type in ["number", "date"]:
            target_number_strs = ["%.3f" % num for num in target_numbers]
            valid_signs_for_add_sub_expressions = self.find_valid_add_sub_expressions(numbers_in_passage,
                                                                                      target_number_strs)
            # ansongn: remove the duplicates caused by assigning a sign to zeros
            zero_idx_list = []
            for i, num in enumerate(numbers_in_passage):
                if num == 0:
                    zero_idx_list.append(i)
            for i in range(len(valid_signs_for_add_sub_expressions)):
                for zero_idx in zero_idx_list:
                    valid_signs_for_add_sub_expressions[i][zero_idx] = 0
            new_valid_expr_list = []
            for expr in valid_signs_for_add_sub_expressions:
                if expr not in new_valid_expr_list:
                    new_valid_expr_list.append(expr)
            valid_signs_for_add_sub_expressions = new_valid_expr_list

        if answer_type in ["number"]:
            # Currently we only support count number 0 ~ 9
            numbers_for_count = list(range(10))
            valid_counts = self.find_valid_counts(numbers_for_count, target_numbers)

        # jferguson Done
        type_to_answer_map = {"passage_span": valid_passage_spans, "question_span": valid_question_spans,
                              "addition_subtraction": valid_signs_for_add_sub_expressions, "counting": valid_counts,
                              "none": answer_type == "none",
                              "binary": answer_type == "binary"}

        if allow_skipping and self.skip_when_all_empty and not any(
                type_to_answer_map[skip_type] for skip_type in self.skip_when_all_empty):
            # print("Skipping: ")
            # print(answer_type)
            # print(question_text)
            # print(context_text)
            # print(answer_dict)
            # print(question_tokens)
            return None

        # jferguson Done
        if answer_type == "binary":
            binary_val = 1 if "yes" in answer_texts else 0
        else:
            binary_val = -1

        answer_info = {"answer_texts": answer_texts,  # this `answer_texts` will not be used for evaluation
                       "answer_passage_spans": valid_passage_spans, "answer_question_spans": valid_question_spans,
                       "signs_for_add_sub_expressions": valid_signs_for_add_sub_expressions, "counts": valid_counts,
                       "none": answer_type == "none", "binary": binary_val}

        return self.qa_text_to_fields(question_tokens, passage_tokens, numbers_as_tokens, passage_number_indices,
                                   passage_number_order, question_number_order, question_number_indices,
                                   answer_info,
                                   additional_metadata={"original_passage": context_text,
                                                        "passage_token_offsets": passage_offset,
                                                        "original_question": question_text,
                                                        "question_token_offsets": question_offset,
                                                        "original_numbers": numbers_in_passage,
                                                        "answer_info": answer_info,
                                                        "answer_annotations": [answer_dict],
                                                        "answer_type": answer_type})

    def qa_text_to_fields(self, question_tokens: List[Token], passage_tokens: List[Token],
                       numbers_as_tokens: List[str], passage_number_indices: List[int],
                       passage_number_order: np.array, question_number_order: np.array,
                       question_number_indices: List[int], answer_info: Dict[str, Any],
                       additional_metadata: Dict[str, Any]) -> Dict[str, Field]:

        fields = dict()
        qc_tokens = [self._tokenizer_start_token] + question_tokens + \
                    [self._tokenizer_sep_token] + passage_tokens + [self._tokenizer_end_token]
        fields['question_with_context'] = TextField(qc_tokens, self._token_indexers)
        fields['question_mask'] = ArrayField(np.array([0]+[1]*len(question_tokens)+[0]+[0]*len(passage_tokens)+[0]))
        fields['passage_mask'] = ArrayField(np.array([0]+[0]*len(question_tokens)+[0]+[1]*len(passage_tokens)+[0]))

        # fields['all_numbers'] = ArrayField(numbers_as_tokens)
        fields['passage_number_order'] = ArrayField(passage_number_order, padding_value=-1)
        fields['question_number_order'] = ArrayField(question_number_order, padding_value=-1)
        # because we appended the question tokens
        concat_passage_number_indices = [(n + len(question_tokens) if n > 0 else n) for n in passage_number_indices]
        fields['passage_number_indices'] = ArrayField(np.array(concat_passage_number_indices), padding_value=-1)
        fields['question_number_indices'] = ArrayField(np.array(question_number_indices), padding_value=-1)

        # fields["ans_texts"] = ArrayField(answer_info["answer_texts"])
        def spans_to_array_field(spans, offset: int = 0):
            if len(spans) > 0:
                return ArrayField(np.array([[a[0]+offset, a[1]+offset] for a in spans]), padding_value=-1)
            else:
                return ArrayField(np.array([[-1, -1]]), padding_value=-1)
        fields["answer_as_passage_spans"] = spans_to_array_field(answer_info["answer_passage_spans"],
                                                                 offset=len(question_tokens)+2)
        fields["answer_as_question_spans"] = spans_to_array_field(answer_info["answer_question_spans"])

        # some preprocessing for add_sub_expression field
        max_num_len = max(1, len(passage_number_indices))
        max_sign_choice = max(1, len(answer_info["signs_for_add_sub_expressions"]))
        answer_as_add_sub_expressions = torch.LongTensor(max_sign_choice, max_num_len).fill_(0)
        sign_len = min(len(answer_info["signs_for_add_sub_expressions"]), max_sign_choice)
        pn_len = len(passage_number_indices) - 1
        for j in range(sign_len):
            answer_as_add_sub_expressions[j, :pn_len] = torch.LongTensor(
                answer_info["signs_for_add_sub_expressions"][j][:pn_len])
        fields["answer_as_add_sub_expressions"] = ArrayField(answer_as_add_sub_expressions.numpy())

        # preprocessing for answer_as_counts
        answer_as_counts = [-1] if len(answer_info['counts']) <= 0 else [answer_info['counts'][0]]
        fields["answer_as_counts"] = ArrayField(np.array(answer_as_counts), padding_value=-1)
        fields["answer_as_none"] = ArrayField(np.array(answer_info["none"]))
        fields["answer_as_binary"] = ArrayField(np.array(answer_info["binary"]))
        fields['span_num'] = ArrayField(np.array([min(8, len(answer_info['answer_texts']))]))

        # add all other metadata
        additional_metadata['question_tokens'] = [token.text for token in question_tokens]
        additional_metadata['passage_tokens'] = [token.text for token in passage_tokens]
        additional_metadata['number_indices'] = passage_number_indices
        additional_metadata['question_id'] = 'iirc_tmp_id'
        fields['metadata'] = MetadataField(additional_metadata)

        return fields

    @staticmethod
    def extract_answer_info_from_annotation(answer_annotation: Dict[str, Any]) -> Tuple[str, List[str]]:
        answer_type = None
        if answer_annotation["spans"]:
            # jferguson Done
            if answer_annotation["spans"] == ["yes"] or answer_annotation["spans"] == ["no"]:
                answer_type = "binary"
            else:
                answer_type = "spans"
        elif answer_annotation["number"]:
            answer_type = "number"
        elif any(answer_annotation["date"].values()):
            answer_type = "date"
        else:
            answer_type = "none"

        answer_content = None
        if answer_type != "none":
            # jferguson Done
            if answer_type == "binary":
                answer_content = answer_annotation["spans"]
            else:
                answer_content = answer_annotation[answer_type] if answer_type is not None else None

        answer_texts: List[str] = []
        if answer_type is None:  # No answer
            pass
        elif answer_type == "none":
            pass
        # jferguson Done
        elif answer_type == "binary":
            answer_texts = answer_content
        elif answer_type == "spans":
            # answer_content is a list of string in this case
            answer_texts = answer_content
        elif answer_type == "date":
            # answer_content is a dict with "month", "day", "year" as the keys
            date_tokens = [answer_content[key] for key in ["month", "day", "year"] if
                           key in answer_content and answer_content[key]]
            answer_texts = date_tokens
        elif answer_type == "number":
            # answer_content is a string of number
            answer_texts = [answer_content]
        return answer_type, answer_texts

    @staticmethod
    def find_valid_spans(passage_tokens: List[Token], answer_texts: List[str]) -> List[Tuple[int, int]]:
        normalized_tokens = [token.text.strip(USTRIPPED_CHARACTERS) for token in passage_tokens]
        # normalized_tokens = passage_tokens
        word_positions: Dict[str, List[int]] = defaultdict(list)
        for i, token in enumerate(normalized_tokens):
            word_positions[token].append(i)
        spans = []
        for answer_text in answer_texts:
            answer_tokens = [token.strip(USTRIPPED_CHARACTERS) for token in answer_text.split()]
            num_answer_tokens = len(answer_tokens)
            if answer_tokens[0] not in word_positions:
                continue
            for span_start in word_positions[answer_tokens[0]]:
                span_end = span_start  # span_end is _inclusive_
                answer_index = 1
                while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                    token = normalized_tokens[span_end + 1]
                    if answer_tokens[answer_index] == token:
                        answer_index += 1
                        span_end += 1
                    # TODO: not sure what this does
                    elif token in IGNORED_TOKENS:
                        span_end += 1
                    else:
                        break
                if num_answer_tokens == answer_index:
                    spans.append((span_start, span_end))
        return spans

    @staticmethod
    def find_valid_add_sub_expressions(numbers: List, targets: List, max_number_of_numbers_to_consider: int = 3) -> \
            List[List[int]]:
        valid_signs_for_add_sub_expressions = []
        # TODO: Try smaller numbers?
        for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
            possible_signs = list(itertools.product((-1, 1), repeat=number_of_numbers_to_consider))
            for number_combination in itertools.combinations(enumerate(numbers), number_of_numbers_to_consider):
                indices = [it[0] for it in number_combination]
                values = [it[1] for it in number_combination]
                for signs in possible_signs:
                    eval_value = sum(sign * value for sign, value in zip(signs, values))
                    # if eval_value in targets:
                    eval_value_str = '%.3f' % eval_value
                    if eval_value_str in targets:
                        labels_for_numbers = [0] * len(numbers)  # 0 represents ``not included''.
                        for index, sign in zip(indices, signs):
                            labels_for_numbers[index] = 1 if sign == 1 else 2  # 1 for positive, 2 for negative
                        valid_signs_for_add_sub_expressions.append(labels_for_numbers)
        return valid_signs_for_add_sub_expressions

    @staticmethod
    def find_valid_counts(count_numbers: List[int], targets: List[int]) -> List[int]:
        valid_indices = []
        for index, number in enumerate(count_numbers):
            if number in targets:
                valid_indices.append(index)
        return valid_indices


def span_sentences(sents: List[Dict], left_idx: int, right_idx: int) -> str:
    result = ''
    for sent in sents:
        s, e = sent['start_idx'], sent['end_idx']
        if s <= left_idx <= right_idx < e:
            return sent['text'][(left_idx - s): (right_idx - s)]
        elif s <= left_idx < e:
            result += sent['text'][left_idx - s:]
        elif s <= right_idx < e:
            result += sent['text'][:right_idx - s]

    return result


def get_number_from_word(word: str, improve_number_extraction=True) -> Union[int, float, None]:
    punctuation = string.punctuation.replace('-', '')
    word = word.strip(punctuation)
    word = word.replace(",", "")
    try:
        number = word_to_num(word)
    except ValueError:
        try:
            number = int(word)
        except ValueError:
            try:
                number = float(word)
            except ValueError:
                if improve_number_extraction:
                    if re.match('^\d*1st$', word):  # ending in '1st'
                        number = int(word[:-2])
                    elif re.match('^\d*2nd$', word):  # ending in '2nd'
                        number = int(word[:-2])
                    elif re.match('^\d*3rd$', word):  # ending in '3rd'
                        number = int(word[:-2])
                    elif re.match('^\d+th$', word):  # ending in <digits>th
                        # Many occurrences are when referring to centuries (e.g "the *19th* century")
                        number = int(word[:-2])
                    elif len(word) > 1 and word[-2] == '0' and re.match('^\d+s$', word):
                        # Decades, e.g. "1960s".
                        # Other sequences of digits ending with s (there are 39 of these in the training
                        # set), do not seem to be arithmetically related, as they are usually proper
                        # names, like model numbers.
                        number = int(word[:-1])
                    elif len(word) > 4 and re.match('^\d+(\.?\d+)?/km[²2]$', word):
                        # per square kilometer, e.g "73/km²" or "3057.4/km2"
                        if '.' in word:
                            number = float(word[:-4])
                        else:
                            number = int(word[:-4])
                    elif len(word) > 6 and re.match('^\d+(\.?\d+)?/month$', word):
                        # per month, e.g "1050.95/month"
                        if '.' in word:
                            number = float(word[:-6])
                        else:
                            number = int(word[:-6])
                    else:
                        return None
                else:
                    return None
    return number


def is_whitespace(c: str) -> bool:
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def roberta_tokenize(text: str, tokenizer, is_answer=False) -> \
        Tuple[List[Token], List[Tuple[int, int]], List[Union[int, float]], List[int], List[int]]:
    split_tokens: List[Token] = []
    sub_token_offsets = []

    numbers = []
    number_indices = []
    number_len = []

    # char_to_word_offset = []
    word_to_char_offset = []
    prev_is_whitespace = True
    tokens = []
    for i, c in enumerate(text):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        elif c in ["-", "–", "~"]:
            tokens.append(c)
            word_to_char_offset.append(i)
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
                word_to_char_offset.append(i)
            else:
                tokens[-1] += c
            prev_is_whitespace = False

    for i, token in enumerate(tokens):
        index = word_to_char_offset[i]
        if i != 0 or is_answer:
            sub_tokens = [a for a in tokenizer.tokenize(" " + token)]
        else:
            sub_tokens = [a for a in tokenizer.tokenize(token)]
        token_number = get_number_from_word(token)

        if token_number is not None:
            numbers.append(token_number)
            number_indices.append(len(split_tokens))
            number_len.append(len(sub_tokens))

        for sub_token in sub_tokens:
            split_tokens.append(sub_token)
            sub_token_offsets.append((index, index + len(token)))

    assert len(split_tokens) == len(sub_token_offsets)
    return split_tokens, sub_token_offsets, numbers, number_indices, number_len


def clipped_passage_num(number_indices: List[int], number_len: List[int],
                        numbers_in_passage: List[int], plen: int) -> Tuple[List[int], List[int], List[int]]:
    if number_indices[-1] < plen:
        return number_indices, number_len, numbers_in_passage
    lo = 0
    hi = len(number_indices) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if number_indices[mid] < plen:
            lo = mid + 1
        else:
            hi = mid
    if number_indices[lo - 1] + number_len[lo - 1] > plen:
        number_len[lo - 1] = plen - number_indices[lo - 1]
    return number_indices[:lo], number_len[:lo], numbers_in_passage[:lo]


def get_number_order(numbers: List[Union[int, float]]) -> List[int]:
    if len(numbers) < 1:
        return []
    ordered_idx_list: list = np.argsort(np.array(numbers)).tolist()

    rank = 0
    number_rank = []
    for i, idx in enumerate(ordered_idx_list):
        if i == 0 or numbers[ordered_idx_list[i]] != numbers[ordered_idx_list[i - 1]]:
            rank += 1
        number_rank.append(rank)

    ordered_idx_rank = zip(ordered_idx_list, number_rank)

    final_rank = sorted(ordered_idx_rank, key=lambda x: x[0])
    final_rank = [item[1] for item in final_rank]

    return final_rank


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # text = " ".join(basic_tokenizer.tokenize(text.strip())).strip()
    text = text.strip().lower()
    if not text:
        return []
    tokens = text.split()
    tokens = [token.strip(STRIPPED_CHARACTERS) for token in tokens]
    return tokens


def iirc_answer_to_drop_style(answer_info: Dict):
    answer_num = ""
    answer_spans = []
    a_type = answer_info["type"]
    if a_type == "span":
        answer_spans = [a["text"] for a in answer_info["answer_spans"]]
        answer_num = ""
    elif a_type == "value":
        answer_spans = []
        answer_num = answer_info["answer_value"]
    elif a_type == "binary":
        # this should be removed: done
        # question_text = question_text[:-1] + ", yes or no?"
        answer_spans = [answer_info["answer_value"]]
        answer_num = ""
    elif a_type == "none":
        answer_spans = []
        answer_num = ""
    elif a_type == "bad":
        return None

    answer_dict = {"date": {"day": "", "month": "", "year": ""},
                   "number": answer_num,
                   "spans": answer_spans}

    return answer_dict
