import logging
import torch
import numpy as np
import re
import string

from scipy.optimize import linear_sum_assignment
from typing import Optional, List, Union, Set, Tuple, Dict, Any
from overrides import overrides
from allennlp.training.metrics.metric import Metric


@Metric.register("set-f-measure")
class SetFMeasure(Metric):

    def __init__(self) -> None:
        self.total_count = 0.0
        self.acc_precision = 0.0
        self.acc_recall = 0.0

    def __call__(
            self,
            predictions: torch.Tensor,
            gold: torch.Tensor,
            mask: Optional[torch.BoolTensor] = None,
    ):
        # just to make sure we detach
        predictions = predictions.detach().float()
        gold = gold.detach().float()

        correct_n = torch.sum(predictions * gold, dim=1)
        predict_n = torch.sum(predictions, dim=1)
        gold_n = torch.sum(gold, dim=1)

        # TODO: think about how to deal with "no links to follow" for unanswerable questions

        # no gold links and predict no links is counted as correct
        no_link_mask = (gold_n == 0.0).float()
        no_link_perf = no_link_mask * (predict_n == 0.0).float()
        precision = correct_n / (predict_n + 1e-12) + no_link_perf
        recall = correct_n / (gold_n + 1e-12) + no_link_perf

        self.total_count += float(predictions.shape[0])
        self.acc_precision += float(np.sum(precision.cpu().numpy()))
        self.acc_recall += float(np.sum(recall.cpu().numpy()))

    def get_metric(self, reset: bool = False):
        avg_precision = self.acc_precision / self.total_count
        avg_recall = self.acc_recall / self.total_count
        avg_f1 = 2.0 / ((1.0 / (avg_precision+1e-30)) + (1.0 / (avg_recall+1e-30)))

        if reset:
            self.reset()

        return avg_precision, avg_recall, avg_f1

    @overrides
    def reset(self):
        self.total_count = 0.0
        self.acc_precision = 0.0
        self.acc_recall = 0.0


@Metric.register("drop-em-f1")
class DropEmAndF1(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    """
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    @overrides
    def __call__(self, prediction: Union[str, List], ground_truths: List):
        """
        Parameters
        ----------
        prediction: ``Union[str, List]``
            The predicted answer from the model evaluated. This could be a string, or a list of string
            when multiple spans are predicted as answer.
        ground_truths: ``List``
            All the ground truth answer annotations.
        """
        # If you wanted to split this out by answer type, you could look at [1] here and group by
        # that, instead of only keeping [0].
        ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in ground_truths]
        # print(f'prediction is {prediction} and ground truth is {ground_truth_answer_strings}')
        exact_match, f1_score = metric_max_over_ground_truths(
            get_drop_metrics,
            prediction,
            ground_truth_answer_strings
        )
        # print(f'exact match score is {exact_match} and f1 score is {f1_score}')
        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official DROP script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __str__(self):
        return "DropEmAndF1(em=%s, f1=%s)" %(self._total_em, self._total_f1)


# From here through _normalize_answer was originally copied from:
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
# Then cleaned up and modified a bit.
def _remove_articles(text: str) -> str:
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)

def _white_space_fix(text: str) -> str:
    return ' '.join(text.split())

EXCLUDE = set(string.punctuation)
def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return ''.join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text

def _lower(text: str) -> str:
    return text.lower()

def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)

def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [_white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
             for token in _tokenize(text)]
    parts = [part for part in parts if part.strip()]
    normalized = ' '.join(parts).strip()
    return normalized

def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False

def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _answer_to_bags(answer: Union[str, List[str], Tuple[str, ...]]) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_drop_metrics(predicted: Union[str, List[str], Tuple[str, ...]],
                gold: Union[str, List[str], Tuple[str, ...]]) -> Tuple[float, float]:
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def answer_json_to_strings(answer: Dict[str, Any]) -> Tuple[Tuple[str, ...], str]:
    """
    Takes an answer JSON blob from the DROP data release and converts it into strings used for
    evaluation.
    """
    if "number" in answer and answer["number"]:
        return tuple([str(answer["number"])]), "number"
    elif "spans" in answer and answer["spans"]:
        return tuple(answer["spans"]), "span" if len(answer["spans"]) == 1 else "spans"
    elif "date" in answer and any(answer["date"].values()):
        return tuple(["{0} {1} {2}".format(answer["date"]["day"],
                                           answer["date"]["month"],
                                           answer["date"]["year"])]), "date"
    else:
        return tuple(["none"]), "none"


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

