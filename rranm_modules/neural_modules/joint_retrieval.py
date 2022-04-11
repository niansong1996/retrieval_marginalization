import logging
from typing import Any, Dict, List, Optional, Set
from overrides import overrides

import itertools
import torch

from torch import nn

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.training.metrics.average import Average

from .link_predictor import TransformerLinkPredictor
from .context_retriever import TransformerContextRetriever


from rranm_modules.utils.iirc_metric import SetFMeasure


logger = logging.getLogger('joint-retriever')


@Model.register('joint-retriever')
class TransformerJointRetriever(Model):
    def __init__(self, vocab: Vocabulary, transformer_model_name: str = "bert-base-cased",
                 beam_size_link: int = 5, beam_size_context: int = 5,
                 print_trajectory: bool = False, load_model_weights: bool = False,
                 link_predictor_weights_file: str = None,
                 context_retriever_weights_file: str = None,
                 use_joint_prob: bool = False,
                 dev_output_file: str = None, second_best_when_null: bool = False,
                 retrieve_sents_per_link: int = 1,
                 **kwargs):
        super().__init__(vocab, **kwargs)
        self._text_field_embedder = BasicTextFieldEmbedder(
            {"tokens": PretrainedTransformerEmbedder(transformer_model_name)}
        )

        self.beam_size_link = beam_size_link
        self.beam_size_context = beam_size_context
        self.print_trajectory = print_trajectory
        self.load_model_weights = load_model_weights
        self.use_joint_prob = use_joint_prob

        self.link_predictor = TransformerLinkPredictor(vocab,
                                                       loaded_transformer_embedder=self._text_field_embedder,
                                                       top_k_eval=self.beam_size_link,
                                                       print_trajectory=self.print_trajectory)
        self.context_retriever = TransformerContextRetriever(vocab,
                                                             loaded_transformer_embedder=self._text_field_embedder,
                                                             loop_during_inference=True,
                                                             print_trajectory=self.print_trajectory,
                                                             dev_output_file=dev_output_file,
                                                             second_best_when_null=second_best_when_null,
                                                             retrieve_sents_per_link=retrieve_sents_per_link)

        if self.load_model_weights:
            self.link_predictor.load_state_dict(torch.load(link_predictor_weights_file))
            self.context_retriever.load_state_dict(torch.load(context_retriever_weights_file))

        self._loss = nn.NLLLoss()

        self.joint_loss = Average()
        self._joint_prf = SetFMeasure()

        self._link_prediction_recall = Average()
        self._joint_retrieval_recall = Average()

    def forward(self, question_with_context: Dict[str, torch.LongTensor],
                all_links_mask: torch.Tensor, gold_links_mask: torch.Tensor,
                context_sents_list: Dict[str, Dict[str, torch.LongTensor]],
                sent_indices_list: torch.Tensor,
                correct_context_mask_list: torch.Tensor,
                # following are meta data fields
                link_pos_name_dict: List[Dict[int, str]],
                gold_link_names: List[Set[str]],
                link_name_idx_dict: List[Dict[str, int]],
                question_text: List[str],
                raw_sents_list: List[List[List[str]]],
                context_title_text_dict: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        # get the batch size
        batch_size = all_links_mask.shape[0]

        # apply the link prediction model but only use the output and not the loss
        link_prediction_output_dict = self.link_predictor(question_with_context, all_links_mask, gold_links_mask,
                                                          question_text, link_pos_name_dict, gold_link_names)
        link_loss = link_prediction_output_dict['loss']
        top_k_link_name_probs = link_prediction_output_dict['top_k_link_name_probs']
        predicted_link_name_list = [list(map(lambda x: x[0], y)) for y in top_k_link_name_probs]

        # covert the link position in the mask to the index of the link in the pre-converted context list
        top_k_link_idx_probs = [list(map(lambda x: (link_name_idx_dict[i][x[0]], x[1]), name_prob_list))
                                for i, name_prob_list in enumerate(top_k_link_name_probs)]
        top_k_indices = [list(map(lambda x: (i, x[0]), v))
                         for i, v in enumerate(top_k_link_idx_probs)]
        top_k_indices_flat = list(itertools.chain.from_iterable(top_k_indices))
        top_k_link_probs = torch.stack([torch.stack(list(map(lambda x: x[1], ins))) for ins in top_k_link_idx_probs], dim=0)

        # further convert to the context chunks of the linked top-k documents
        token_ids = torch.stack([context_sents_list['tokens']['token_ids'][i, j] for i, j in top_k_indices_flat], dim=0)
        mask = torch.stack([context_sents_list['tokens']['mask'][i, j] for i, j in top_k_indices_flat], dim=0)
        type_ids = torch.stack([context_sents_list['tokens']['type_ids'][i, j] for i, j in top_k_indices_flat], dim=0)
        sent_indices = torch.stack([sent_indices_list[i, j] for i, j in top_k_indices_flat], dim=0)
        correct_context_mask = torch.stack([correct_context_mask_list[i, j] for i, j in top_k_indices_flat], dim=0)

        # shape: (batch_size, beam_size_link, n_chunks, max_seq_length/1/None)
        token_ids = token_ids.view(batch_size, -1, token_ids.shape[1], token_ids.shape[2])
        mask = mask.view(batch_size, -1, mask.shape[1], mask.shape[2])
        type_ids = type_ids.view(batch_size, -1, type_ids.shape[1], type_ids.shape[2])
        sent_indices = sent_indices.view(batch_size, -1, sent_indices.shape[1], sent_indices.shape[2])
        correct_context_mask = correct_context_mask.view(batch_size, -1, correct_context_mask.shape[1])
        selected_raw_sents_list = [[] for _ in range(batch_size)]
        for i, j in top_k_indices_flat:
            selected_raw_sents_list[i].append(raw_sents_list[i][j])

        context_sents = {'tokens': {'token_ids': token_ids, 'mask': mask, 'type_ids': type_ids}}

        # compute the context scores
        context_output_dict = self.context_retriever(context_sents, sent_indices, correct_context_mask,
                                                     predicted_link_name_list, question_text, selected_raw_sents_list,
                                                     context_title_text_dict,
                                                     link_probs=top_k_link_probs if self.use_joint_prob else None)
        context_probs = context_output_dict['context_probs']
        context_loss = context_output_dict['loss']

        # # compute the joint probability
        # top_k_probs = top_k_probs.reshape(-1).unsqueeze(dim=-1).expand_as(context_probs)
        # joint_probs = top_k_probs * context_probs
        #
        # # compute the loss
        # gold_indices = torch.argmax(correct_context_mask, dim=1)
        # loss = self._loss(torch.log(joint_probs), gold_indices)
        #
        # # evaluation
        # self.joint_loss(float(loss))

        output_dict = {'loss': 0.5 * link_loss + (1 - 0.5) * context_loss}
        # output_dict = {'loss': link_loss}
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        joint_retrieval_metrics = dict()

        link_prediction_metrics = self.link_predictor.get_metrics(reset)
        context_retrieval_metric = self.context_retriever.get_metrics(reset)
        joint_retrieval_metrics.update(link_prediction_metrics)
        joint_retrieval_metrics.update(context_retrieval_metric)

        return joint_retrieval_metrics

