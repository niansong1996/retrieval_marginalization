import logging
from typing import Any, Dict, List, Optional, Set
from math import log

import torch

from torch import nn

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, Average
from allennlp.common.util import sanitize_wordpiece
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.nn.util import get_token_ids_from_text_field_tensors

from rranm_modules.utils.tokenizer_utils import replace_masked_values_with_big_negative_number
from rranm_modules.utils.iirc_metric import SetFMeasure
from rranm_modules.neural_modules.numnet_utils import replace_masked_values

from rranm_modules.neural_modules.base_model import TransformerBasedModel


@Model.register('link-predictor')
class TransformerLinkPredictor(TransformerBasedModel):
    def __init__(self, vocab: Vocabulary, loaded_transformer_embedder: BasicTextFieldEmbedder = None,
                 transformer_model_name: str = "bert-base-cased", top_k_eval: int = 3,
                 print_trajectory: bool = False, **kwargs):
        super().__init__(vocab, loaded_transformer_embedder, transformer_model_name, print_trajectory, **kwargs)

        self.top_k_eval = top_k_eval

        self._linear_layer = nn.Linear(self._text_field_embedder.get_output_dim(), 1)

        self._prf_metric = SetFMeasure()
        self._link_prediction_recall = Average()
        self._loss_eval = Average()
        self._link_num_loss_eval = Average()
        self._selected_link_num = Average()

    def forward(self, question_with_context: Dict[str, torch.LongTensor],
                all_links_mask: torch.Tensor, gold_links_mask: torch.Tensor,
                # following are metadata fields
                question_text: List[str],
                link_pos_name_dict: List[Dict[int, str]],
                gold_link_names: List[Set[str]],
                **kwargs) -> Dict[str, torch.Tensor]:
        # some sizes
        batch_size = all_links_mask.shape[0]

        # replace_masked_values_with_big_negative_number(...) function only supports boolean types
        all_links_mask = all_links_mask.bool()
        non_gold_links_mask = all_links_mask.float() - gold_links_mask.float()

        embedded_question = self._text_field_embedder(question_with_context)
        link_logits = self._linear_layer(embedded_question).squeeze(2)
        link_probs = replace_masked_values(torch.sigmoid(link_logits), all_links_mask, 1e-7)

        # compute the NLLLoss (log prob of the gold link set)
        gold_links_log_prob = gold_links_mask * torch.log(link_probs)
        non_gold_link_log_prob = non_gold_links_mask * torch.log(1.0-link_probs)
        link_set_log_prob = torch.sum(gold_links_log_prob, dim=-1) + torch.sum(non_gold_link_log_prob, dim=-1)
        loss = -1.0 * link_set_log_prob.mean()
        link_num_loss = torch.square(torch.sum(link_probs, dim=1) - torch.sum(gold_links_mask, dim=1)).mean()
        self._link_num_loss_eval(float(link_num_loss))
        self._loss_eval(float(loss))

        # evaluations
        selected_link_indices = torch.nonzero(torch.where(link_probs > 0.5, link_probs,
                                                          torch.zeros_like(link_probs)), as_tuple=False)
        selected_mask = torch.zeros_like(link_probs)
        for idx in selected_link_indices:
            selected_mask[idx[0]][idx[1]] = 1.0
        self._prf_metric(selected_mask, gold_links_mask)

        # compute the log prob of the selected set
        selected_link_set_log_prob = torch.sum(selected_mask*torch.log(link_probs), dim=-1) \
                                           + torch.sum((all_links_mask.float()-selected_mask) *
                                                       torch.log(1.0-link_probs), dim=-1)

        # for each instance, a list of tuples of (link_pos, link_prob)
        link_probs_sparse = replace_masked_values(link_probs, all_links_mask, 0.0).to_sparse()
        indices = link_probs_sparse.indices().transpose(0, 1)
        link_pos_probs = [[] for _ in range(link_probs.shape[0])]
        for i in range(indices.shape[0]):
            t = indices[i]
            link_pos_probs[int(t[0])].append((int(t[1]), link_probs_sparse.values()[i]))

        # get rid of duplicated linked targets with the highest link score
        link_name_probs_dict_list = [dict(map(lambda x: (link_pos_name_dict[i][x[0]], x[1]),
                                          sorted(link_pos_prob, key=lambda y: y[1])))
                                     for i, link_pos_prob in enumerate(link_pos_probs)]
        # TODO handling the case where the # of links < k (min # links across training set is 6)
        selected_link_name_probs = [sorted(name_probs_dict.items(), key=lambda x: x[1], reverse=True)[:self.top_k_eval]
                                 for name_probs_dict in link_name_probs_dict_list]

        selected_link_num = [0] * batch_size
        for i, name_probs_tuple_list in enumerate(selected_link_name_probs):
            for j, nam_probs_tuple in enumerate(name_probs_tuple_list):
                if nam_probs_tuple[1] < 0.5:
                    selected_link_name_probs[i][j] = ('null context', torch.tensor([0.0])[0])
                else:
                    selected_link_num[i] += 1
        self._selected_link_num(sum(selected_link_num) / batch_size)

        # get the link prediction trajectory
        output_json_list = []
        for i, link_name_prob_pairs in enumerate(selected_link_name_probs):
            self.log_trajectory('[question] {}'.format(question_text[i]))
            link_name_prob_dict = dict(link_name_prob_pairs)
            predict_link_name_set = set(link_name_prob_dict.keys())

            gold_link_name_set = gold_link_names[i]
            link_prediction_traj = '[link-prediction] example-{} top-{} link probabilities: {}, gold links are {}' \
                .format(i, self.top_k_eval, link_name_prob_dict, gold_link_name_set)
            self.log_trajectory(link_prediction_traj)

            link_prediction_top_k_recall = (len(predict_link_name_set.intersection(gold_link_name_set)) + 1e-12) / \
                                           (len(gold_link_name_set) + 1e-12)
            self._link_prediction_recall(link_prediction_top_k_recall)
            self.log_trajectory('[link-prediction] top-{} recall is [{}]'
                                .format(self.top_k_eval, link_prediction_top_k_recall))

            output_json = {'predicted_links': list(filter(lambda x: x != 'null context', predict_link_name_set)),
                           'gold_question_links': list(filter(lambda x: x != 'null context', gold_link_name_set))}
            output_json_list.append(output_json)

        output_dict = {'loss': loss+link_num_loss, 'top_k_link_name_probs': selected_link_name_probs,
                       'predicted_link_set_log_prob': selected_link_set_log_prob,
                       'output_json_list': output_json_list}

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        p, r, f = self._prf_metric.get_metric(reset)
        lp_top_k_recall = self._link_prediction_recall.get_metric(reset)
        lp_loss = self._loss_eval.get_metric(reset)
        lp_n_loss = self._link_num_loss_eval.get_metric(reset)
        select_link_num = self._selected_link_num.get_metric(reset)
        return {'_lp_p': p, '_lp_r': r, '_lp_f': f, 'lp_n': select_link_num,
                'lp_recall': lp_top_k_recall, 'lp_n_loss': lp_n_loss, 'lp_loss': lp_loss}
