import torch
import torch.nn as nn
import numpy as np

from typing import List, Tuple, Dict, Union, Any

import torch.nn.functional as F
import allennlp.nn.util as util
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.metrics.average import Average

from rranm_modules.neural_modules.base_model import TransformerBasedModel
from rranm_modules.utils.iirc_metric import DropEmAndF1
# from allennlp_models.rc.metrics.drop_em_and_f1 import DropEmAndF1
from rranm_modules.neural_modules.numnet_utils import FFNLayer, GCN, ResidualGRU
from rranm_modules.neural_modules.numnet_utils import get_best_span, best_answers_extraction, convert_number_to_str
from rranm_modules.neural_modules.numnet_utils import NumNetTextFieldEmbedder, NumNetTransformerEmbedder
from rranm_modules.neural_modules.numnet_utils import replace_masked_values


@Model.register('numnet-qa')
class NumnetQA(TransformerBasedModel):
    def __init__(self, vocab: Vocabulary, hidden_size: int,
                 loaded_transformer_embedder: BasicTextFieldEmbedder = None,
                 transformer_model_name: str = "bert-base-cased", print_trajectory: bool = False,
                 dropout_prob: float = 0.1,
                 answering_abilities: List[str] = None,
                 use_gcn: bool = False,
                 gcn_steps: int = 1,
                 answer_types: List[str] = None) -> None:

        # use designated version of the BERT to output every hidden layer's state
        if loaded_transformer_embedder is None:
            loaded_transformer_embedder = NumNetTextFieldEmbedder(
                {"tokens": NumNetTransformerEmbedder(transformer_model_name)})

        super().__init__(vocab, loaded_transformer_embedder, transformer_model_name, print_trajectory)
        self.use_gcn = use_gcn
        self.bert = self._text_field_embedder
        modeling_out_dim = hidden_size
        if answering_abilities is None:
            self.answering_abilities = ["passage_span_extraction", "question_span_extraction",
                                        "addition_subtraction", "counting", "none", "binary"]
        else:
            self.answering_abilities = answering_abilities

        if answer_types is None:
            self.answer_types = ["none", "binary", "spans", "date", "number"]
        else:
            self.answer_types = answer_types

        self._drop_metrics = DropEmAndF1()
        self._drop_metrics_by_answer_type = [DropEmAndF1() for _ in self.answer_types]
        self._percentage_by_type = [Average() for _ in self.answer_types]

        if len(self.answering_abilities) > 1:
            self._answer_ability_predictor = FFNLayer(3 * hidden_size, hidden_size, len(self.answering_abilities),
                                                      dropout_prob)

        if "passage_span_extraction" in self.answering_abilities or "question_span_extraction" in self.answering_abilities:
            self._passage_span_extraction_index = self.answering_abilities.index("passage_span_extraction")
            self._question_span_extraction_index = self.answering_abilities.index("question_span_extraction")
            self._span_start_predictor = nn.Linear(4 * hidden_size, 1, bias=False)
            self._span_end_predictor = nn.Linear(4 * hidden_size, 1, bias=False)

        if "addition_subtraction" in self.answering_abilities:
            self._addition_subtraction_index = self.answering_abilities.index("addition_subtraction")
            self._number_sign_predictor = FFNLayer(5 * hidden_size, hidden_size, 3, dropout_prob)

        if "counting" in self.answering_abilities:
            self._counting_index = self.answering_abilities.index("counting")
            self._count_number_predictor = FFNLayer(5 * hidden_size, hidden_size, 10, dropout_prob)

        if "binary" in self.answering_abilities:
            self._binary_answer_predictor = FFNLayer(4 * hidden_size, hidden_size, 2, dropout_prob)

        self._dropout = torch.nn.Dropout(p=dropout_prob)

        if self.use_gcn:
            node_dim = modeling_out_dim

            self._gcn_input_proj = nn.Linear(node_dim * 2, node_dim)
            self._gcn = GCN(node_dim=node_dim, iteration_steps=gcn_steps)
            self._iteration_steps = gcn_steps
            self._proj_ln = nn.LayerNorm(node_dim)
            self._proj_ln0 = nn.LayerNorm(node_dim)
            self._proj_ln1 = nn.LayerNorm(node_dim)
            self._proj_ln3 = nn.LayerNorm(node_dim)
            self._gcn_enc = ResidualGRU(hidden_size, dropout_prob, 2)
        # add bert proj
        self._proj_sequence_h = nn.Linear(hidden_size, 1, bias=False)
        self._proj_number = nn.Linear(hidden_size * 2, 1, bias=False)

        self._proj_sequence_g0 = FFNLayer(hidden_size, hidden_size, 1, dropout_prob)
        self._proj_sequence_g1 = FFNLayer(hidden_size, hidden_size, 1, dropout_prob)
        self._proj_sequence_g2 = FFNLayer(hidden_size, hidden_size, 1, dropout_prob)

        # span num extraction
        self._proj_span_num = FFNLayer(3 * hidden_size, hidden_size, 9, dropout_prob)

    def forward(self,  # type: ignore
                question_with_context: Dict[str, torch.LongTensor],
                passage_mask: torch.LongTensor,
                question_mask: torch.LongTensor,
                passage_number_indices: torch.LongTensor,
                question_number_indices: torch.LongTensor,
                passage_number_order: torch.LongTensor,
                question_number_order: torch.LongTensor,
                answer_as_passage_spans: torch.Tensor = None,
                answer_as_question_spans: torch.Tensor = None,
                answer_as_add_sub_expressions: torch.Tensor = None,
                answer_as_counts: torch.LongTensor = None,
                answer_as_none: torch.FloatTensor = None,
                answer_as_binary: torch.FloatTensor = None,
                span_num: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        # TODO: figure out why allennlp's arraryField can't have 2-D array
        # answer_as_add_sub_expressions = torch.from_numpy(np.array(answer_as_add_sub_expressions))
        # metadata = None

        # the huggingface roberta output: (last_hidden_layer, pooler_output, hidden_states, attentions)
        outputs = self._text_field_embedder(question_with_context, only_keep_last_layer_output=False)

        sequence_output = outputs[0]
        sequence_output_list = [item for item in outputs[2][-4:]]

        batch_size = passage_mask.shape[0]
        if self.use_gcn and \
           ("passage_span_extraction" in self.answering_abilities or "question_span" in self.answering_abilities):
            # M2, M3
            sequence_alg = self._gcn_input_proj(torch.cat([sequence_output_list[2], sequence_output_list[3]], dim=2))
            encoded_passage_for_numbers = sequence_alg
            encoded_question_for_numbers = sequence_alg
            # passage number extraction
            real_number_indices = passage_number_indices - 1
            number_mask = (real_number_indices > -1).bool()  # ??
            clamped_number_indices = replace_masked_values(real_number_indices, number_mask, 0)
            encoded_numbers = torch.gather(encoded_passage_for_numbers, 1,
                                           clamped_number_indices.unsqueeze(-1).expand(-1, -1,
                                                                                       encoded_passage_for_numbers.size(
                                                                                           -1)).long())

            # question number extraction
            question_number_mask = (question_number_indices > -1).bool()
            clamped_question_number_indices = replace_masked_values(question_number_indices, question_number_mask,
                                                                         0)
            question_encoded_number = torch.gather(encoded_question_for_numbers, 1,
                                                   clamped_question_number_indices.unsqueeze(-1).expand(-1, -1,
                                                                                                        encoded_question_for_numbers.size(
                                                                                                            -1)).long())

            # graph mask
            number_order = torch.cat((passage_number_order, question_number_order), -1)
            new_graph_mask = number_order.unsqueeze(1).expand(batch_size, number_order.size(-1),
                                                              -1) > number_order.unsqueeze(-1).expand(batch_size, -1,
                                                                                                      number_order.size(
                                                                                                          -1))
            new_graph_mask = new_graph_mask.long()
            all_number_mask = torch.cat((number_mask, question_number_mask), dim=-1)
            new_graph_mask = all_number_mask.unsqueeze(1) * all_number_mask.unsqueeze(-1) * new_graph_mask

            # iteration
            d_node, q_node, d_node_weight, _ = self._gcn(d_node=encoded_numbers, q_node=question_encoded_number,
                                                         d_node_mask=number_mask, q_node_mask=question_number_mask,
                                                         graph=new_graph_mask)
            gcn_info_vec = torch.zeros((batch_size, sequence_alg.size(1) + 1, sequence_output_list[-1].size(-1)),
                                       dtype=torch.float, device=d_node.device)
            clamped_number_indices = replace_masked_values(real_number_indices, number_mask,
                                                                gcn_info_vec.size(1) - 1)
            gcn_info_vec.scatter_(1, clamped_number_indices.unsqueeze(-1).expand(-1, -1, d_node.size(-1)).long(), d_node)
            gcn_info_vec = gcn_info_vec[:, :-1, :]

            sequence_output_list[2] = self._gcn_enc(self._proj_ln(sequence_output_list[2] + gcn_info_vec))
            sequence_output_list[0] = self._gcn_enc(self._proj_ln0(sequence_output_list[0] + gcn_info_vec))
            sequence_output_list[1] = self._gcn_enc(self._proj_ln1(sequence_output_list[1] + gcn_info_vec))
            sequence_output_list[3] = self._gcn_enc(self._proj_ln3(sequence_output_list[3] + gcn_info_vec))

        # passage hidden and question hidden
        sequence_h2_weight = self._proj_sequence_h(sequence_output_list[2]).squeeze(-1)
        passage_h2_weight = util.masked_softmax(sequence_h2_weight, passage_mask)
        passage_h2 = util.weighted_sum(sequence_output_list[2], passage_h2_weight)
        question_h2_weight = util.masked_softmax(sequence_h2_weight, question_mask)
        question_h2 = util.weighted_sum(sequence_output_list[2], question_h2_weight)

        # passage g0, g1, g2
        question_g0_weight = self._proj_sequence_g0(sequence_output_list[0]).squeeze(-1)
        question_g0_weight = util.masked_softmax(question_g0_weight, question_mask)
        question_g0 = util.weighted_sum(sequence_output_list[0], question_g0_weight)

        question_g1_weight = self._proj_sequence_g1(sequence_output_list[1]).squeeze(-1)
        question_g1_weight = util.masked_softmax(question_g1_weight, question_mask)
        question_g1 = util.weighted_sum(sequence_output_list[1], question_g1_weight)

        question_g2_weight = self._proj_sequence_g2(sequence_output_list[2]).squeeze(-1)
        question_g2_weight = util.masked_softmax(question_g2_weight, question_mask)
        question_g2 = util.weighted_sum(sequence_output_list[2], question_g2_weight)

        if len(self.answering_abilities) > 1:
            # Shape: (batch_size, number_of_abilities)
            answer_ability_logits = self._answer_ability_predictor(
                torch.cat([passage_h2, question_h2, sequence_output[:, 0]], 1))
            answer_ability_log_probs = F.log_softmax(answer_ability_logits, -1)
            best_answer_ability = torch.argmax(answer_ability_log_probs, 1)

        real_number_indices = passage_number_indices.squeeze(-1) - 1
        number_mask = (real_number_indices > -1).bool()
        clamped_number_indices = replace_masked_values(real_number_indices, number_mask, 0)
        encoded_passage_for_numbers = torch.cat([sequence_output_list[2], sequence_output_list[3]], dim=-1)
        encoded_numbers = torch.gather(encoded_passage_for_numbers, 1,
                                       clamped_number_indices.unsqueeze(-1).expand(-1, -1,
                                                                                   encoded_passage_for_numbers.size(
                                                                                       -1)).long())
        number_weight = self._proj_number(encoded_numbers).squeeze(-1)
        number_mask = (passage_number_indices > -1).bool()
        number_weight = util.masked_softmax(number_weight, number_mask)
        number_vector = util.weighted_sum(encoded_numbers, number_weight)

        if "counting" in self.answering_abilities:
            # Shape: (batch_size, 10)
            count_number_logits = self._count_number_predictor(
                torch.cat([number_vector, passage_h2, question_h2, sequence_output[:, 0]], dim=1))
            count_number_log_probs = torch.nn.functional.log_softmax(count_number_logits, -1)
            # Info about the best count number prediction
            # Shape: (batch_size,)
            best_count_number = torch.argmax(count_number_log_probs, -1)
            best_count_log_prob = torch.gather(count_number_log_probs, 1, best_count_number.unsqueeze(-1).long()).squeeze(-1)
            if len(self.answering_abilities) > 1:
                best_count_log_prob += answer_ability_log_probs[:, self._counting_index]

        if "passage_span_extraction" in self.answering_abilities or "question_span_extraction" in self.answering_abilities:
            # start 0, 2
            sequence_for_span_start = torch.cat([sequence_output_list[2],
                                                 sequence_output_list[0],
                                                 sequence_output_list[2] * question_g2.unsqueeze(1),
                                                 sequence_output_list[0] * question_g0.unsqueeze(1)],
                                                dim=2)
            sequence_span_start_logits = self._span_start_predictor(sequence_for_span_start).squeeze(-1)
            # Shape: (batch_size, passage_length, modeling_dim * 2)
            sequence_for_span_end = torch.cat([sequence_output_list[2],
                                               sequence_output_list[1],
                                               sequence_output_list[2] * question_g2.unsqueeze(1),
                                               sequence_output_list[1] * question_g1.unsqueeze(1)],
                                              dim=2)
            # Shape: (batch_size, passage_length)
            sequence_span_end_logits = self._span_end_predictor(sequence_for_span_end).squeeze(-1)
            # Shape: (batch_size, passage_length)

            # span number prediction
            span_num_logits = self._proj_span_num(torch.cat([passage_h2, question_h2, sequence_output[:, 0]], dim=1))
            span_num_log_probs = torch.nn.functional.log_softmax(span_num_logits, -1)

            best_span_number = torch.argmax(span_num_log_probs, dim=-1)

            if "passage_span_extraction" in self.answering_abilities:
                passage_span_start_log_probs = util.masked_log_softmax(sequence_span_start_logits, passage_mask)
                passage_span_end_log_probs = util.masked_log_softmax(sequence_span_end_logits, passage_mask)

                # Info about the best passage span prediction
                passage_span_start_logits = replace_masked_values(sequence_span_start_logits, passage_mask, -1e7)
                passage_span_end_logits = replace_masked_values(sequence_span_end_logits, passage_mask, -1e7)
                # Shage: (batch_size, topk, 2)
                best_passage_span = get_best_span(passage_span_start_logits, passage_span_end_logits)

            if "question_span_extraction" in self.answering_abilities:
                question_span_start_log_probs = util.masked_log_softmax(sequence_span_start_logits, question_mask)
                question_span_end_log_probs = util.masked_log_softmax(sequence_span_end_logits, question_mask)

                # Info about the best question span prediction
                question_span_start_logits = replace_masked_values(sequence_span_start_logits, question_mask, -1e7)
                question_span_end_logits = replace_masked_values(sequence_span_end_logits, question_mask, -1e7)
                # Shape: (batch_size, topk, 2)
                best_question_span = get_best_span(question_span_start_logits, question_span_end_logits)

        if "addition_subtraction" in self.answering_abilities:
            alg_encoded_numbers = torch.cat(
                [encoded_numbers,
                 question_h2.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                 passage_h2.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                 sequence_output[:, 0].unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)
                 ], 2)

            # Shape: (batch_size, # of numbers in the passage, 3)
            number_sign_logits = self._number_sign_predictor(alg_encoded_numbers)
            number_sign_log_probs = torch.nn.functional.log_softmax(number_sign_logits, -1)

            # Shape: (batch_size, # of numbers in passage).
            best_signs_for_numbers = torch.argmax(number_sign_log_probs, -1)
            # For padding numbers, the best sign masked as 0 (not included).
            best_signs_for_numbers = replace_masked_values(best_signs_for_numbers, number_mask, 0)
            # Shape: (batch_size, # of numbers in passage)
            best_signs_log_probs = torch.gather(number_sign_log_probs, 2, best_signs_for_numbers.unsqueeze(-1).long()).squeeze(
                -1)
            # the probs of the masked positions should be 1 so that it will not affect the joint probability
            # TODO: this is not quite right, since if there are many numbers in the passage,
            # TODO: the joint probability would be very small.
            best_signs_log_probs = replace_masked_values(best_signs_log_probs, number_mask, 0)
            # Shape: (batch_size,)
            best_combination_log_prob = best_signs_log_probs.sum(-1)
            if len(self.answering_abilities) > 1:
                best_combination_log_prob += answer_ability_log_probs[:, self._addition_subtraction_index]

        if "binary" in self.answering_abilities:
            # jferguson: TODO - add log prob calculation for binary answers
            sequence_for_binary = torch.cat([sequence_output_list[2],
                                             sequence_output_list[0],
                                             sequence_output_list[2] * question_g2.unsqueeze(1),
                                             sequence_output_list[0] * question_g0.unsqueeze(1)],
                                            dim=2)[:, 0]
            binary_logits = self._binary_answer_predictor(sequence_for_binary)
            binary_log_probs = torch.nn.functional.log_softmax(binary_logits, -1)
            _, best_binary_answers = binary_log_probs.max(dim=1)
        output_dict = {}

        # TODO: jferguson - check where these are defined
        # If answer is given, compute the loss.
        if answer_as_passage_spans is not None \
                or answer_as_question_spans is not None \
                or answer_as_add_sub_expressions is not None \
                or answer_as_counts is not None \
                or answer_as_none is not None \
                or answer_as_binary is not None:

            log_marginal_likelihood_list = []

            for answering_ability in self.answering_abilities:
                if answering_ability == "passage_span_extraction":
                    # Shape: (batch_size, # of answer spans)
                    gold_passage_span_starts = answer_as_passage_spans[:, :, 0]
                    gold_passage_span_ends = answer_as_passage_spans[:, :, 1]
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_passage_span_mask = (gold_passage_span_starts != -1).bool()
                    clamped_gold_passage_span_starts = replace_masked_values(gold_passage_span_starts,
                                                                                  gold_passage_span_mask, 0)
                    clamped_gold_passage_span_ends = replace_masked_values(gold_passage_span_ends,
                                                                                gold_passage_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_span_starts = torch.gather(passage_span_start_log_probs, 1,
                                                                          clamped_gold_passage_span_starts.long())
                    log_likelihood_for_passage_span_ends = torch.gather(passage_span_end_log_probs, 1,
                                                                        clamped_gold_passage_span_ends.long())
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_spans = log_likelihood_for_passage_span_starts + log_likelihood_for_passage_span_ends
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_passage_spans = replace_masked_values(log_likelihood_for_passage_spans,
                                                                                  gold_passage_span_mask, -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_passage_span = util.logsumexp(log_likelihood_for_passage_spans)

                    # span log probabilities
                    log_likelihood_for_passage_span_nums = torch.gather(span_num_log_probs, 1, span_num.long())
                    log_likelihood_for_passage_span_nums = replace_masked_values(
                        log_likelihood_for_passage_span_nums,
                        gold_passage_span_mask[:, :1], -1e7)
                    log_marginal_likelihood_for_passage_span_nums = util.logsumexp(log_likelihood_for_passage_span_nums)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span +
                                                        log_marginal_likelihood_for_passage_span_nums)

                elif answering_ability == "question_span_extraction":
                    # Shape: (batch_size, # of answer spans)
                    gold_question_span_starts = answer_as_question_spans[:, :, 0]
                    gold_question_span_ends = answer_as_question_spans[:, :, 1]
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_question_span_mask = (gold_question_span_starts != -1).bool()
                    clamped_gold_question_span_starts = replace_masked_values(gold_question_span_starts,
                                                                                   gold_question_span_mask, 0)
                    clamped_gold_question_span_ends = replace_masked_values(gold_question_span_ends,
                                                                                 gold_question_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_span_starts = torch.gather(question_span_start_log_probs, 1,
                                                                           clamped_gold_question_span_starts.long())
                    log_likelihood_for_question_span_ends = torch.gather(question_span_end_log_probs, 1,
                                                                         clamped_gold_question_span_ends.long())
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_spans = log_likelihood_for_question_span_starts + log_likelihood_for_question_span_ends
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_question_spans = replace_masked_values(log_likelihood_for_question_spans,
                                                                                   gold_question_span_mask, -1e7)
                    # Shape: (batch_size, )
                    # pylint: disable=invalid-name
                    log_marginal_likelihood_for_question_span = util.logsumexp(log_likelihood_for_question_spans)

                    # question multi span prediction
                    log_likelihood_for_question_span_nums = torch.gather(span_num_log_probs, 1, span_num.long())
                    log_marginal_likelihood_for_question_span_nums = util.logsumexp(
                        log_likelihood_for_question_span_nums)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_question_span +
                                                        log_marginal_likelihood_for_question_span_nums)

                elif answering_ability == "addition_subtraction":
                    # The padded add-sub combinations use 0 as the signs for all numbers, and we mask them here.
                    # Shape: (batch_size, # of combinations)
                    gold_add_sub_mask = (answer_as_add_sub_expressions.sum(-1) > 0).float()
                    # Shape: (batch_size, # of numbers in the passage, # of combinations)
                    gold_add_sub_signs = answer_as_add_sub_expressions.transpose(1, 2)
                    # Shape: (batch_size, # of numbers in the passage, # of combinations)
                    log_likelihood_for_number_signs = torch.gather(number_sign_log_probs, 2, gold_add_sub_signs.long())
                    # the log likelihood of the masked positions should be 0
                    # so that it will not affect the joint probability
                    log_likelihood_for_number_signs = replace_masked_values(log_likelihood_for_number_signs,
                                                                                 number_mask.unsqueeze(-1), 0)
                    # Shape: (batch_size, # of combinations)
                    log_likelihood_for_add_subs = log_likelihood_for_number_signs.sum(1)
                    # For those padded combinations, we set their log probabilities to be very small negative value
                    log_likelihood_for_add_subs = replace_masked_values(log_likelihood_for_add_subs,
                                                                             gold_add_sub_mask, -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_add_sub = util.logsumexp(log_likelihood_for_add_subs)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_add_sub)

                elif answering_ability == "counting":
                    # Count answers are padded with label -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    # Shape: (batch_size, # of count answers)
                    gold_count_mask = (answer_as_counts != -1).bool()
                    # Shape: (batch_size, # of count answers)
                    clamped_gold_counts = replace_masked_values(answer_as_counts, gold_count_mask, 0)
                    log_likelihood_for_counts = torch.gather(count_number_log_probs, 1, clamped_gold_counts.long())
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_counts = replace_masked_values(log_likelihood_for_counts, gold_count_mask,
                                                                           -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_count = util.logsumexp(log_likelihood_for_counts)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_count)
                elif answering_ability == "none":
                    # If answer_as_none is 1, then answer is None, and likelihood should be used. otherwise, give 0 by adding -1e7
                    log_marginal_likelihood_list.append((1 - answer_as_none) * -1e7);
                elif answering_ability == "binary":
                    # jferguson append likelihood
                    # (4,2), select according to answer_as_binary if > -1, else -1e7
                    mask = (answer_as_binary != -1)
                    clamped_binary_answers = replace_masked_values(answer_as_binary, mask, 0)
                    log_likelihood_for_binary = torch.gather(binary_log_probs, 1,
                                                             clamped_binary_answers.unsqueeze(1).long()).squeeze(1)
                    log_likelihood_for_binary = replace_masked_values(log_likelihood_for_binary, mask, -1e7)
                    log_marginal_likelihood_list.append(log_likelihood_for_binary)

                else:
                    raise ValueError("Unsupported answering ability: %s" % (answering_ability))
            # print(log_marginal_likelihood_list)
            if len(self.answering_abilities) > 1:
                # Add the ability probabilities if there are more than one abilities
                all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                all_log_marginal_likelihoods = all_log_marginal_likelihoods + answer_ability_log_probs
                marginal_log_likelihood = util.logsumexp(all_log_marginal_likelihoods)
                # assert all(map(lambda x: x<0, marginal_log_likelihood.detach().cpu().numpy())), f'prob>1: \n' \
                #                                                 f'marginal_log_likelihood: \n {marginal_log_likelihood}\n' \
                #                                                 f'all_log_marginal_likelihoods: \n {all_log_marginal_likelihoods}\n' \
                #                                                 f'answer_probs: \n {torch.exp(answer_ability_log_probs)}'
            else:
                marginal_log_likelihood = log_marginal_likelihood_list[0]
            # TODO (ansongn): write a checker on the probabilities
            output_dict["loss"] = - marginal_log_likelihood.mean()
            output_dict["marginal_log_likelihood"] = marginal_log_likelihood

        if metadata is not None:
            output_dict["question_id"] = []
            output_dict["answer"] = []
            output_json_list = []
            for i in range(batch_size):
                if len(self.answering_abilities) > 1:
                    predicted_ability_str = self.answering_abilities[best_answer_ability[i].detach().cpu().numpy()]
                else:
                    predicted_ability_str = self.answering_abilities[0]

                answer_json: Dict[str, Any] = {'predicted_answer_ability': predicted_ability_str}

                question_start = 1
                passage_start = len(metadata[i]["question_tokens"]) + 2
                # We did not consider multi-mention answers here
                if predicted_ability_str == "passage_span_extraction":
                    answer_json["answer_type"] = "passage_span"
                    passage_str = metadata[i]['original_passage']
                    offsets = metadata[i]['passage_token_offsets']
                    predicted_answer, predicted_spans = best_answers_extraction(best_passage_span[i],
                                                                                best_span_number[i], passage_str,
                                                                                offsets, passage_start)
                    answer_json["value"] = predicted_answer
                    answer_json["spans"] = predicted_spans
                elif predicted_ability_str == "question_span_extraction":
                    answer_json["answer_type"] = "question_span"
                    question_str = metadata[i]['original_question']
                    offsets = metadata[i]['question_token_offsets']
                    predicted_answer, predicted_spans = best_answers_extraction(best_question_span[i],
                                                                                best_span_number[i], question_str,
                                                                                offsets, question_start)
                    answer_json["value"] = predicted_answer
                    answer_json["spans"] = predicted_spans
                elif predicted_ability_str == "addition_subtraction":
                    answer_json["answer_type"] = "arithmetic"
                    original_numbers = metadata[i]['original_numbers']
                    sign_remap = {0: 0, 1: 1, 2: -1}
                    predicted_signs = [sign_remap[it] for it in best_signs_for_numbers[i].detach().cpu().numpy()]
                    result = sum([sign * number for sign, number in zip(predicted_signs, original_numbers)])
                    predicted_answer = convert_number_to_str(result)
                    offsets = metadata[i]['passage_token_offsets']
                    number_indices = metadata[i]['number_indices']
                    number_positions = [offsets[index - 1] for index in number_indices]
                    answer_json['numbers'] = []
                    for offset, value, sign in zip(number_positions, original_numbers, predicted_signs):
                        answer_json['numbers'].append({'span': offset, 'value': value, 'sign': sign})
                    if number_indices[-1] == -1:
                        # There is a dummy 0 number at position -1 added in some cases; we are
                        # removing that here.
                        answer_json["numbers"].pop()
                    answer_json["value"] = result
                    answer_json['number_sign_log_probs'] = number_sign_log_probs[i, :, :].detach().cpu().numpy()

                elif predicted_ability_str == "counting":
                    answer_json["answer_type"] = "count"
                    predicted_count = best_count_number[i].detach().cpu().numpy()
                    predicted_answer = str(predicted_count)
                    answer_json["count"] = predicted_count
                # TODO: jferguson - Add in output for answer type NONE
                elif predicted_ability_str == "none":
                    answer_json["answer_type"] = "none"
                    predicted_answer = "none"
                    answer_json["value"] = "none"
                elif predicted_ability_str == "binary":
                    answer_json["answer_type"] = "binary"
                    predicted_answer = "yes" if best_binary_answers[i] == 1 else "no"
                    answer_json["value"] = predicted_answer
                else:
                    raise ValueError("Unsupported answer ability: %s" % (predicted_ability_str))

                answer_json["predicted_answer"] = predicted_answer
                output_dict["question_id"].append(metadata[i]["question_id"])
                output_dict["answer"].append(answer_json)
                answer_annotations = metadata[i].get('answer_annotations', [])
                if answer_annotations:
                    # print(f'predicted type is {predicted_ability_str}, '
                    #       f'actual type is {metadata[i]["answer_type"]}')
                    tmp_em_1, tmp_f1_1 = self._drop_metrics._total_em, self._drop_metrics._total_f1
                    self._drop_metrics(predicted_answer, answer_annotations)
                    tmp_em_2, tmp_f1_2 = self._drop_metrics._total_em, self._drop_metrics._total_f1

                    instance_answer_type = metadata[i]["answer_type"]
                    assert instance_answer_type in self.answer_types

                    output_json_list.append({'predicted_answer_ability': predicted_ability_str,
                                             'predicted_answer': predicted_answer,
                                             'gold_answer_type': instance_answer_type,
                                             'gold_answer': answer_annotations,
                                             'em': float(tmp_em_2 - tmp_em_1),
                                             'f1': float(tmp_f1_2 - tmp_f1_1)})

                    for j, answer_type in enumerate(self.answer_types):
                        if instance_answer_type == answer_type:
                            self._drop_metrics_by_answer_type[j](predicted_answer, answer_annotations)
                            self._percentage_by_type[j](1.0)
                        else:
                            self._percentage_by_type[j](0.0)

            if self.use_gcn:
                output_dict['clamped_number_indices'] = clamped_number_indices
                output_dict['node_weight'] = d_node_weight

            output_dict['output_json_list'] = output_json_list
        # print(f'loss is: {output_dict["loss"]}')
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return_dict = {'qa_em': exact_match, 'qa_f1': f1_score}

        # update the answer-type-specific metrics
        for i, answer_type in enumerate(self.answer_types):
            _, f1_score = self._drop_metrics_by_answer_type[i].get_metric(reset)
            avg_percentage = self._percentage_by_type[i].get_metric(reset)
            return_dict[f'_{answer_type}_f1'] = f1_score
            return_dict[f'_{answer_type}_per'] = avg_percentage

        return return_dict

