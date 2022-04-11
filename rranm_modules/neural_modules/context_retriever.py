import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from torch import nn

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from allennlp.training.metrics import Average

from .base_model import TransformerBasedModel

from rranm_modules.utils.iirc_metric import SetFMeasure
from rranm_modules.utils.tokenizer_utils import replace_masked_values_with_big_negative_number


@Model.register('context-retriever')
class TransformerContextRetriever(TransformerBasedModel):
    def __init__(self, vocab: Vocabulary, loaded_transformer_embedder: BasicTextFieldEmbedder = None,
                 transformer_model_name: str = "bert-base-cased", loop_during_inference: bool = True,
                 print_trajectory: bool = False,
                 dev_output_file: str = None, second_best_when_null: bool = False,
                 retrieve_sents_per_link: int = 1, **kwargs):
        super().__init__(vocab, loaded_transformer_embedder, transformer_model_name, print_trajectory, **kwargs)

        self.loop_during_inference = loop_during_inference

        self.dev_output_file = open(dev_output_file, 'w+') if dev_output_file is not None else None
        self.second_best_when_null = second_best_when_null
        self.retrieve_sents_per_link = retrieve_sents_per_link

        self.training_sents_n = None
        self._linear_layer = nn.Linear(self._text_field_embedder.get_output_dim(), 1)

        self._loss = nn.NLLLoss()

        self._loss_eval = Average()
        self._accuracy = CategoricalAccuracy()
        self._prf = SetFMeasure()
        self._joint_retrieval_recall = Average()

    def get_chunk_embedding_loop(self, context_sents: Dict[str, Dict[str, torch.Tensor]],
                            sent_indices: torch.Tensor):
        # if the model didn't go into training model (e.g. load the model and evaluate), set a safe number
        if self.training_sents_n is None:
            self.training_sents_n = 16

        # merge the batch, link and chunk dimension
        init_shape = context_sents['tokens']['token_ids'].shape
        merged_context_sents = {'tokens': dict()}
        merged_context_sents['tokens']['token_ids'] = context_sents['tokens']['token_ids'] \
            .view(init_shape[0] * init_shape[1] * init_shape[2], -1)
        merged_context_sents['tokens']['mask'] = context_sents['tokens']['mask'] \
            .view(init_shape[0] * init_shape[1] * init_shape[2], -1)
        merged_context_sents['tokens']['type_ids'] = context_sents['tokens']['type_ids'] \
            .view(init_shape[0] * init_shape[1] * init_shape[2], -1)

        # split to individual chunks and encode them separately to improve scalability
        context_sents_list = zip(torch.split(merged_context_sents['tokens']['token_ids'], self.training_sents_n, dim=0),
                                 torch.split(merged_context_sents['tokens']['mask'], self.training_sents_n, dim=0),
                                 torch.split(merged_context_sents['tokens']['type_ids'], self.training_sents_n, dim=0))
        context_sents_list = list(map(lambda x: {'tokens': {'token_ids': x[0],
                                                            'mask': x[1],
                                                            'type_ids': x[2]}}, context_sents_list))
        embedded_context_sents = torch.cat([self._text_field_embedder(context_sent)
                                              for context_sent in context_sents_list], dim=0)

        # pick the embedding at the sentence start position
        sent_indices = sent_indices.view(-1)
        embedded_context_sents = embedded_context_sents[torch.arange(embedded_context_sents.shape[0]), sent_indices]
        embedded_context_sents = embedded_context_sents.view(init_shape[0], init_shape[1], init_shape[2], -1)

        return embedded_context_sents

    def get_chunk_embedding(self, context_sents: Dict[str, Dict[str, torch.Tensor]],
                            sent_indices: torch.Tensor):
        init_shape = context_sents['tokens']['token_ids'].shape
        embedded_context_sents = self._text_field_embedder(context_sents, num_wrapping_dims=2)

        # pick the embedding at the sentence start position
        embedded_context_sents = embedded_context_sents.view(init_shape[0] * init_shape[1] * init_shape[2],
                                                             init_shape[3], -1)
        sent_indices = sent_indices.view(-1)
        embedded_context_sents = embedded_context_sents[torch.arange(embedded_context_sents.shape[0]), sent_indices]
        embedded_context_sents = embedded_context_sents.view(init_shape[0], init_shape[1], init_shape[2], -1)

        return embedded_context_sents

    def forward(self, context_sents: Dict[str, Dict[str, torch.Tensor]],
                sent_indices: torch.Tensor, correct_context_mask: torch.Tensor,
                # following are meta data fields
                link_name_list: List[List[str]],
                question_text: List[str],
                raw_sents_list: List[List[List[str]]],
                context_title_text_dict: List[Dict[str, str]],
                link_probs: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # get the batch size (i.e. number of questions per batch)
        batch_size = context_sents['tokens']['token_ids'].shape[0]
        link_per_question = context_sents['tokens']['token_ids'].shape[1]

        if not self.training and self.loop_during_inference:
            embedded_context_sents = self.get_chunk_embedding_loop(context_sents, sent_indices)
        else:
            if not self.training_sents_n:
                init_shape = context_sents['tokens']['token_ids'].shape
                self.training_sents_n = init_shape[0] * init_shape[1]
            embedded_context_sents = self.get_chunk_embedding(context_sents, sent_indices)

        # pass through the linear layer and compute the probabilities
        logits = self._linear_layer(embedded_context_sents).squeeze(3)
        valid_chunk_mask = context_sents['tokens']['mask'][:, :, :, 0]
        logits = replace_masked_values_with_big_negative_number(logits, valid_chunk_mask)
        context_probs_with_null = torch.softmax(logits, dim=2)

        if link_probs is not None:
            context_probs_with_null = context_probs_with_null * link_probs.unsqueeze(dim=2)

        # evaluations for the accuracy
        correct_context_indices = torch.argmax(correct_context_mask, dim=2)

        predict_context_mask = torch.zeros_like(context_probs_with_null)
        top_chunk_indices = torch.argmax(context_probs_with_null, dim=2)
        for i in range(batch_size):
            for j in range(link_per_question):
                predict_context_mask[i][j][top_chunk_indices[i][j]] = 1.0
        self._accuracy(predict_context_mask, correct_context_indices)
        self._prf(predict_context_mask.reshape(-1, predict_context_mask.shape[2]),
                  correct_context_mask.reshape(-1, correct_context_mask.shape[2]))

        loss = self._loss(torch.log(context_probs_with_null.reshape(-1, context_probs_with_null.shape[2])),
                          correct_context_indices.view(-1))
        self._loss_eval(float(loss))

        # get the context retrieval trajectory
        retrieved_sents = []
        output_json_list = []
        batch_context_probs = context_probs_with_null.detach()
        for i, link_context_probs in enumerate(batch_context_probs):
            total_gold_context_n = len(context_title_text_dict[i])
            output_json = {'gold_link_name_sent_list':
                           list(map(lambda x: {'title': x[0], 'sent': x[1]}, context_title_text_dict[i].items())),
                           'predicted_link_name_sent_list': []}
            non_null_good_context_n = 0
            for j, sent_probs in enumerate(link_context_probs):
                link_name = link_name_list[i][j]
                retrieved_sents_for_link = []
                max_sent_prob_idx = int(torch.argmax(sent_probs))
                # assert max_sent_prob_idx == int(torch.argsort(sent_probs, descending=True)[0]), \
                #     '{} {} {} {}'.format(max_sent_prob_idx,
                #                          sent_probs[max_sent_prob_idx],
                #                          int(torch.argsort(sent_probs, descending=True)[0]),
                #                          sent_probs[int(torch.argsort(sent_probs, descending=True)[0])])
                correct_sent_idx = int(torch.argmax(correct_context_mask[i][j]))
                correct_sent_text = raw_sents_list[i][j][correct_sent_idx]
                context_retrival_traj = '[context-retrieval] example-{} link-{} max sent prob [{}]: \"{}\",' \
                                        'correct sent: \"{}\"' \
                    .format(i, link_name,  float(sent_probs[max_sent_prob_idx]),
                            raw_sents_list[i][j][max_sent_prob_idx], correct_sent_text)
                self.log_trajectory(context_retrival_traj)

                if link_name != 'null context':
                    output_json['predicted_link_name_sent_list'].append({'title': link_name,
                                                                         'sent': raw_sents_list[i][j][max_sent_prob_idx]})

                for idx in torch.argsort(sent_probs, descending=True):
                    if len(raw_sents_list[i][j]) <= int(idx):
                        break
                    retrieved_sent = raw_sents_list[i][j][int(idx)]
                    if retrieved_sent.lower() == 'null' and self.second_best_when_null:
                        continue

                    retrieved_sents_for_link.append(int(idx))
                    retrieved_sents.append(raw_sents_list[i][j][int(idx)])
                    if len(retrieved_sents_for_link) == self.retrieve_sents_per_link:
                        break

                # calculate question level recall on context
                if correct_sent_idx in retrieved_sents_for_link and correct_sent_text.strip().lower() != 'null':
                    non_null_good_context_n += 1
            self.log_trajectory('[joint-retrieval] gold evidence title-text dict is {}'
                                .format(context_title_text_dict[i]))
            output_json_list.append(output_json)
            question_level_recall = (non_null_good_context_n + 1e-12) / (total_gold_context_n + 1e-12)
            self.log_trajectory('[joint-retrieval] question-level recall [{}]'.format(question_level_recall))
            self._joint_retrieval_recall(question_level_recall)

        output_dict = {'loss': loss, 'context_probs': context_probs_with_null,
                       'retrieved_sents': retrieved_sents, 'output_json_list': output_json_list}

        if self.dev_output_file is not None:
            self.dev_output_file.write('{} @@@ {}\n'.format(question_text[0].replace('\n', ''),
                                                            ' ### '.join(retrieved_sents).replace('\n', '')))
            self.dev_output_file.flush()

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy = self._accuracy.get_metric(reset)
        p, r, f = self._prf.get_metric(reset)
        cr_loss = self._loss_eval.get_metric(reset)
        joint_recall = self._joint_retrieval_recall.get_metric(reset)
        return {'cr_acc': accuracy, '_cr_f': f, '_cr_loss': cr_loss, 'jr_recall': joint_recall}
