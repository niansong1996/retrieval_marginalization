import logging
import allennlp
from typing import Any, Dict, List, Optional, Set, Union
from overrides import overrides

import itertools
import json
import torch
import allennlp.nn.util as util

from torch import nn
from functools import reduce
from os import path

from allennlp.data import Vocabulary, Instance
from allennlp.data.batch import Batch
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.training.metrics.average import Average
from allennlp.nn.util import move_to_device
from rranm_modules.utils.iirc_metric import DropEmAndF1

from .link_predictor import TransformerLinkPredictor
from .context_retriever import TransformerContextRetriever
from .numnet_qa import NumnetQA
from ..readers.iirc_qa_reader import IIRCQAReader
from rranm_modules.neural_modules.numnet_utils import NumNetTextFieldEmbedder, NumNetTransformerEmbedder


from rranm_modules.utils.iirc_metric import SetFMeasure


logger = logging.getLogger('joint-qa')


@Model.register('joint-qa')
class TransformerJointQA(Model):
    def __init__(self, vocab: Vocabulary, transformer_model_name: str = "bert-base-cased",
                 # below are the fields for retrieval
                 beam_size_link: int = 5,
                 print_trajectory: bool = False, load_model_weights: bool = False,
                 link_predictor_weights_file: str = None,
                 context_retriever_weights_file: str = None,
                 use_joint_prob: bool = False,
                 dev_output_file: str = None, second_best_when_null: bool = False,
                 retrieve_sents_per_link: int = 1,
                 gold_link_for_retrieval_training: bool = False,
                 # below are the fields for initializing qa reader (how to create qa instance out of retrieved context)
                 skip_when_all_empty: List[str] = None,
                 relaxed_span_match_for_finding_labels: bool = True,
                 q_max_tokens: int = 64,
                 c_max_tokens: int = 463,
                 # below are the fields for initializing qa reasoner (how to do qa base on the created instance)
                 hidden_size: int = 768,
                 dropout_prob: float = 0.1, answering_abilities: List[str] = None,
                 use_gcn: bool = False, gcn_steps: int = 1,
                 # below are some fields specific to the joint qa modeling
                 supervised_link_loss_weight: float = 1.0,
                 supervised_context_loss_weight: float = 1.0,
                 top_m_context: int = 4,
                 marginalization_loss_weight: float = 1.0,
                 gold_context_loss_weight: float = 0.0,
                 invalid_context_loss_weight: float = 0.0,
                 use_link_prediction_model: bool = True,
                 use_context_retrieval_model: bool = True,
                 use_qa_model: bool = True,
                 save_prediction_path: Union[str, None] = None,
                 eval_with_marginalization: bool = False,
                 saved_lp_model_path: Union[str, None] = None,
                 saved_cr_model_path: Union[str, None] = None,
                 saved_qa_model_path: Union[str, None] = None,
                 **kwargs):
        super().__init__(vocab, **kwargs)

        # as for bert, we have to use a customized version for all because QA-model rely on all layer's output
        # from the roberta model
        self.transformer_model_name = transformer_model_name
        self._text_field_embedder = NumNetTextFieldEmbedder(
            {"tokens": NumNetTransformerEmbedder(self.transformer_model_name)})

        # params for retrieval
        self.beam_size_link = beam_size_link
        self.print_trajectory = print_trajectory
        self.load_model_weights = load_model_weights
        self.use_joint_prob = use_joint_prob
        self.gold_link_for_retrieval_training = gold_link_for_retrieval_training

        # params for qa
        self.supervised_link_loss_weight = supervised_link_loss_weight
        self.supervised_context_loss_weight = supervised_context_loss_weight
        self.top_m_context = top_m_context
        self.marginalization_loss_weight = marginalization_loss_weight
        self.gold_context_loss_weight = gold_context_loss_weight
        self.invalid_context_loss_weight = invalid_context_loss_weight
        self.use_lp_model = use_link_prediction_model
        self.use_cr_model = use_context_retrieval_model
        self.use_qa_model = use_qa_model
        self.eval_with_gold_context_only = not self.use_lp_model and not self.use_cr_model \
                                           and self.use_qa_model and self.top_m_context == 0
        self.eval_with_marginalization = eval_with_marginalization

        self.saved_lp_model_path = saved_lp_model_path
        self.saved_cr_model_path = saved_cr_model_path
        self.saved_qa_model_path = saved_qa_model_path

        if self.eval_with_marginalization:
            self._marginalization_qa_eval = DropEmAndF1()

        self.save_prediction_path = save_prediction_path
        if self.save_prediction_path is not None and path.exists(self.save_prediction_path):
            raise ValueError(f'{save_prediction_path} already exists!')

        # for qa-reasoner, build a simplified qa_reader because we need to generate instances on the fly
        self.qa_reader = IIRCQAReader('', skip_when_all_empty, relaxed_span_match_for_finding_labels,
                                      q_max_tokens=q_max_tokens, c_max_tokens=c_max_tokens,
                                      no_loading_wiki_dict=True, transformer_model_name=transformer_model_name)

        # initializing all three sub-modules
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
        self.qa_reasoner = NumnetQA(vocab, hidden_size,
                                    loaded_transformer_embedder=self._text_field_embedder,
                                    transformer_model_name=transformer_model_name,
                                    print_trajectory=self.print_trajectory,
                                    dropout_prob=dropout_prob,
                                    answering_abilities=answering_abilities,
                                    use_gcn=use_gcn,
                                    gcn_steps=gcn_steps)

        self.module_weights_loaded = False

        # FIXME: this currently has some issue: can't load two individually trained models and combine them
        if self.load_model_weights:
            self.link_predictor.load_state_dict(torch.load(link_predictor_weights_file))
            self.context_retriever.load_state_dict(torch.load(context_retriever_weights_file))

        self._loss = nn.NLLLoss()

        self._qa_marginalization_loss = Average()
        self._qa_gold_context_loss = Average()
        self._qa_invalid_context_loss = Average()
        self._marginalization_set_size = Average()
        self._qa_loss = Average()

    @staticmethod
    def load_bert_weights_for_module(whole_model: torch.nn.Module, bert_name: str, module: torch.nn.Module,
                                     saved_model_path: str, module_name: str, device: torch.device):


        # first make sure the bert models are not shared anymore
        module._text_field_embedder = NumNetTextFieldEmbedder(
            {"tokens": NumNetTransformerEmbedder(bert_name)})

        param_dict = {}
        param_dict = torch.load(saved_model_path, map_location=device)
        param_dict = dict(filter(lambda x: x[0].startswith(module_name), param_dict.items()))
        overlapping_params = set(whole_model.state_dict().keys()).intersection(param_dict.keys())
        logger.info(f"loading {list(set(map(lambda x: x.split('.')[0], list(param_dict.keys()))))}, "
                    f"{len(overlapping_params)} params got updated!")

        whole_model.load_state_dict(param_dict, strict=False)
        module.to(device)

    def load_module_weights(self, device: torch.device):
        logger.info(f'device is {device}')
        if self.saved_lp_model_path:
            self.load_bert_weights_for_module(self, self.transformer_model_name, self.link_predictor,
                                              self.saved_lp_model_path, 'link_predictor', device)
        if self.saved_cr_model_path:
            self.load_bert_weights_for_module(self, self.transformer_model_name, self.context_retriever,
                                              self.saved_cr_model_path, 'context_retriever', device)
        if self.saved_qa_model_path:
            self.load_bert_weights_for_module(self, self.transformer_model_name, self.qa_reasoner,
                                              self.saved_qa_model_path, 'qa_reasoner', device)
        if self.saved_lp_model_path and self.saved_cr_model_path and self.saved_qa_model_path:
            del self._text_field_embedder
        self.module_weights_loaded = True

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
                context_title_text_dict: List[Dict[str, str]],
                # added the following for the qa part
                answer_dict: List[Dict[str, Any]],
                original_paragraph: List[str],
                ) -> Dict[str, torch.Tensor]:

        if not self.module_weights_loaded:
            self.load_module_weights(all_links_mask.device)

        # get the batch size
        batch_size = all_links_mask.shape[0]

        link_loss = 0.0
        if self.use_lp_model:
            # apply the link prediction model but only use the output and not the loss
            link_prediction_output_dict = self.link_predictor(question_with_context, all_links_mask, gold_links_mask,
                                                              question_text, link_pos_name_dict, gold_link_names)
            link_loss = link_prediction_output_dict['loss']
            predicted_link_set_prob = link_prediction_output_dict['predicted_link_set_log_prob']
            top_k_link_name_probs = link_prediction_output_dict['top_k_link_name_probs']
            lp_output_json_list = link_prediction_output_dict['output_json_list']

            if not (self.use_cr_model or self.use_qa_model):
                return {'loss': link_loss}

        if (self.gold_link_for_retrieval_training and self.training) or (not self.use_lp_model):
            top_k_link_name_probs = [[] for _ in gold_link_names]
            for i, ins_gold_link_set in enumerate(gold_link_names):
                for j, gold_link_name in enumerate(ins_gold_link_set):
                    if j < self.beam_size_link:
                        top_k_link_name_probs[i].append((gold_link_name, torch.tensor([0.0])[0]))
                while len(top_k_link_name_probs[i]) < self.beam_size_link:
                    top_k_link_name_probs[i].append(('null context', torch.tensor([0.0])[0]))

        top_k_link_name_probs = move_to_device(top_k_link_name_probs, all_links_mask.device)

        context_loss = 0.0
        if self.use_cr_model:
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
            cr_output_json_list = context_output_dict['output_json_list']

            if not self.use_qa_model:
                return {'loss': self.supervised_link_loss_weight * link_loss +
                                self.supervised_context_loss_weight * context_loss}
        else:
            context_probs = torch.zeros((batch_size, self.beam_size_link, len(raw_sents_list[0][0])),
                                        device=all_links_mask.device)

        if self.training or self.eval_with_gold_context_only or self.eval_with_marginalization:
            marginalization_set_size = self.top_m_context
        else:
            marginalization_set_size = 1

        # formulate the set for marginalization
        # for faster computation, for each of the link, only keep the top-m sents, with their original index
        context_log_probs = torch.log(context_probs)
        context_log_probs, context_indices = torch.topk(context_log_probs, marginalization_set_size, dim=2)

        link_num = context_log_probs.shape[1]
        sent_per_link = context_log_probs.shape[2]

        # get the prob for each combination of sents as context and rank them by joint prob
        context_sents_prob_list = [get_tryout_combinations([sent_per_link]*link_num)
                                   for _ in range(batch_size)]
        for i in range(batch_size):
            for sent_idx_comb in context_sents_prob_list[i]:
                comb_log_prob = 0.0
                for j in range(link_num):
                    comb_log_prob += context_log_probs[i][j][sent_idx_comb[j]]
                sent_idx_comb.append(comb_log_prob)

        context_sents_prob_list = list(map(lambda y:
                                           sorted(y, key=lambda x: x[link_num], reverse=True)[:marginalization_set_size],
                                           context_sents_prob_list))

        # create the qa instances based on the retrieved context
        qa_instances_probs = [[] for _ in range(len(context_sents_prob_list))]
        for idx_in_batch, indices_prob in enumerate(context_sents_prob_list):

            for i in range(len(indices_prob)):
                context_prob = indices_prob[i][-1]
                sent_text_list = []
                for j in range(len(indices_prob[i])-1):
                    actual_sent_idx = context_indices[idx_in_batch][j][indices_prob[i][j]]
                    sent_text = selected_raw_sents_list[idx_in_batch][j][actual_sent_idx]
                    if sent_text != 'NULL':
                        sent_text_list.append(sent_text)

                sent_text_list.append(original_paragraph[idx_in_batch])
                context = (" " + self.qa_reader._tokenizer_sep_token.text + " ").join(sent_text_list)

                if self.training:
                    qa_fields = self.qa_reader.get_qa_fields(question_text[idx_in_batch], context,
                                                             answer_dict[idx_in_batch], allow_skipping=True)
                else:
                    qa_fields = self.qa_reader.get_qa_fields(question_text[idx_in_batch], context,
                                                             answer_dict[idx_in_batch], allow_skipping=False)

                if qa_fields is not None:
                    qa_instances_probs[idx_in_batch].append((Instance(qa_fields),
                                                             context_prob+predicted_link_set_prob[idx_in_batch]))
                elif self.training:
                    dummy_answer_dict = {'date': {'day': '', 'month': '', 'year': ''}, 'number': '', 'spans': []}
                    qa_fields = self.qa_reader.get_qa_fields(question_text[idx_in_batch], context, dummy_answer_dict,
                                                             allow_skipping=True)
                    assert qa_fields is not None
                    qa_instances_probs[idx_in_batch].append((Instance(qa_fields),
                                                             torch.ones(1, device=context_probs.device)[0]))
                else:
                    raise ValueError('This should not happen: in evaluation mode, qa fields is None')

        if self.training or self.eval_with_gold_context_only:
            # manually add the gold context
            for i in range(batch_size):
                gold_sents = []
                for gold_link_name in gold_link_names[i]:
                    if gold_link_name in link_name_idx_dict[i]:
                        gold_link_idx = link_name_idx_dict[i][gold_link_name]
                        gold_sent_idx = int(correct_context_mask_list[i][gold_link_idx].argmax(dim=0))
                        gold_sent = raw_sents_list[i][gold_link_idx][gold_sent_idx]
                        if gold_sent != "NULL":
                            gold_sents.append(gold_sent)
                    else:
                        # this happens because sometimes, the links questions annotators specified as useful can not
                        # even be found in the passage (so this is marked as unanswerable)
                        continue

                gold_sents.append(original_paragraph[i])
                context = (" " + self.qa_reader._tokenizer_sep_token.text + " ").join(gold_sents)

                qa_fields = self.qa_reader.get_qa_fields(question_text[i], context, answer_dict[i])

                if qa_fields is not None:
                    qa_instances_probs[i].append((Instance(qa_fields), torch.zeros(1, device=context_probs.device)[0]))
                else:
                    logger.info('skipping: still can\'t derive gold answer even if this is gold context!!!')

        # all the examples will be formed into one batch so they can be passed into the qa model at once
        qa_instances_num_in_batch = list(map(lambda x: len(x), qa_instances_probs))
        valid_qa_ins_prob_tuple = list(filter(lambda x: x[0] is not None, reduce(lambda y, z: y+z, qa_instances_probs)))

        if len(valid_qa_ins_prob_tuple) > 0:
            qa_batch_instances = list(map(lambda x: x[0], valid_qa_ins_prob_tuple))
            qa_batch_instances_context_probs = list(map(lambda x: x[1], valid_qa_ins_prob_tuple))


            qa_batch = Batch(qa_batch_instances)
            qa_batch.index_instances(self.vocab)
            qa_batch_padding_lengths = qa_batch.get_padding_lengths()
            qa_batch_input = qa_batch.as_tensor_dict(qa_batch_padding_lengths)
            qa_batch_input = move_to_device(qa_batch_input, context_probs.device)

            qa_batch_output_dict = self.qa_reasoner(**qa_batch_input)
            qa_gold_answer_log_probs = qa_batch_output_dict['marginal_log_likelihood']
            qa_output_json_list = qa_batch_output_dict['output_json_list']

            # divide things back to each example, and figure out marginalization part and extra part
            # three losses represents the marginalization loss, the added gold context loss, and the invalid context loss
            qa_marginalization_set_log_prob_list = [[] for _ in range(batch_size)]
            qa_gold_context_log_prob_list = []
            qa_invalid_context_log_prob_list = [[] for _ in range(batch_size)]
            predicted_answer_context_prob_list = [[] for _ in range(batch_size)]
            start = 0
            for j, qa_instances_num in enumerate(qa_instances_num_in_batch):
                qa_instances_answer_log_probs = qa_gold_answer_log_probs[start:start+qa_instances_num]
                qa_instances_context_log_probs = qa_batch_instances_context_probs[start:start+qa_instances_num]
                qa_instances_output_jsons = qa_output_json_list[start:start+qa_instances_num]
                for i in range(qa_instances_num):
                    if qa_instances_context_log_probs[i] == 0.0:
                        # this is manually added gold context instances
                        qa_gold_context_log_prob_list.append(qa_instances_answer_log_probs[i])
                    elif qa_instances_context_log_probs[i] == 1.0:
                        # this is extra loss on invalid context instances
                        qa_invalid_context_log_prob_list[j].append(qa_instances_answer_log_probs[i])
                    elif qa_instances_context_log_probs[i] < 0.0:
                        # this should be included in the marginalization
                        qa_marginalization_set_log_prob_list[j]\
                            .append(qa_instances_context_log_probs[i] + qa_instances_answer_log_probs[i])
                        predicted_answer_context_prob_list[j].append((qa_instances_output_jsons[i],
                                                                      qa_instances_context_log_probs[i]))
                    else:
                        raise ValueError(f'Context prob {float(qa_instances_context_log_probs[i])} is not defined')
                start += qa_instances_num
            # compute the marginalization loss
            marginalization_loss = 0.0
            for log_prob_list in qa_marginalization_set_log_prob_list:
                if len(log_prob_list) > 0:
                    marginalization_loss += -1.0 * util.logsumexp(torch.stack(log_prob_list, dim=0))
            marginalization_loss /= batch_size

            # compute the invalid context loss
            invalid_context_loss = 0.0
            for log_prob_list in qa_invalid_context_log_prob_list:
                if len(log_prob_list) > 0:
                    invalid_context_loss += -1.0 * sum(log_prob_list)
            invalid_context_loss /= batch_size

            # compute the gold context loss
            gold_context_loss = 0.0
            if len(qa_gold_context_log_prob_list) > 0:
                gold_context_loss += -1.0 * sum(qa_gold_context_log_prob_list) / batch_size
            qa_loss = self.marginalization_loss_weight * marginalization_loss \
                      + self.gold_context_loss_weight * gold_context_loss \
                      + self.invalid_context_loss_weight * invalid_context_loss

            self._qa_marginalization_loss(float(marginalization_loss))
            self._qa_gold_context_loss(float(gold_context_loss))
            self._qa_invalid_context_loss(float(invalid_context_loss))
            self._qa_loss(float(qa_loss))
            self._marginalization_set_size(sum(map(lambda x: len(x), qa_marginalization_set_log_prob_list)) / batch_size)

            if self.eval_with_marginalization:
                # aggregate the answers with context prob over the marginalization set
                for j, marg_set in enumerate(predicted_answer_context_prob_list):
                    answer_prob_list_dict: Dict[str, float] = dict()
                    for qa_output_json, context_prob in marg_set:
                        pred_answer = qa_output_json['predicted_answer']
                        pred_answer = pred_answer[0] if isinstance(pred_answer, list) else pred_answer
                        predicted_answer_str = str(pred_answer)
                        context_prob_float = float(torch.exp(context_prob))
                        if predicted_answer_str not in answer_prob_list_dict:
                            answer_prob_list_dict[predicted_answer_str] = context_prob_float
                        else:
                            answer_prob_list_dict[predicted_answer_str] += context_prob_float
                    agg_answer_probs = sorted(answer_prob_list_dict.items(), key=lambda x: x[1], reverse=True)
                    best_answer = agg_answer_probs[0][0]
                    self._marginalization_qa_eval(best_answer, marg_set[0][0]['gold_answer'])
        else:
            # if even the gold retrieved sent can't derive the gold answer by NumNet,
            # then only use it to train retrieval
            qa_loss = 0.0

        if (not self.training) and self.use_lp_model and self.use_cr_model \
                and self.use_qa_model and self.save_prediction_path:
            assert len(lp_output_json_list) == len(cr_output_json_list) == len(qa_output_json_list)
            for i, output_json_fields in enumerate(zip(lp_output_json_list, cr_output_json_list, qa_output_json_list)):
                json_output = {'question': question_text[i],
                               'original_paragraph': original_paragraph[i],
                               'link_prediction': output_json_fields[0],
                               'context_retrieval': output_json_fields[1],
                               'qa': output_json_fields[2]}

                str_output = json.dumps(json_output)
                with open(self.save_prediction_path, 'a') as f:
                    f.write(str_output + '\n')

        loss = self.supervised_link_loss_weight * link_loss \
               + self.supervised_context_loss_weight * context_loss \
               + qa_loss
        output_dict = {'loss': loss}
        # output_dict = {'loss': link_loss}
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, Any]:
        joint_qa_metrics = dict()

        link_prediction_metrics = self.link_predictor.get_metrics(reset) if self.use_lp_model else {}
        context_retrieval_metric = self.context_retriever.get_metrics(reset) if self.use_cr_model else {}
        qa_metric = self.qa_reasoner.get_metrics(reset) if self.use_qa_model else {}

        joint_qa_metrics.update(link_prediction_metrics)
        joint_qa_metrics.update(context_retrieval_metric)
        joint_qa_metrics.update(qa_metric)

        joint_qa_metrics['qa_marg_size'] = self._marginalization_set_size.get_metric(reset)
        joint_qa_metrics['qa_marg_loss'] = self._qa_marginalization_loss.get_metric(reset)
        joint_qa_metrics['qa_gold_c_loss'] = self._qa_gold_context_loss.get_metric(reset)
        joint_qa_metrics['qa_invalid_c_loss'] = self._qa_invalid_context_loss.get_metric(reset)
        joint_qa_metrics['qa_loss'] = self._qa_loss.get_metric(reset)

        if self.eval_with_marginalization:
            em, f1 = self._marginalization_qa_eval.get_metric(reset)
            joint_qa_metrics['marg_eval_qa_em'] = em
            joint_qa_metrics['marg_eval_qa_f1'] = f1

        return joint_qa_metrics


def get_tryout_combinations(sizes: List[int]):
    if len(sizes) == 0:
        return [[]]

    multipliers = []
    for i in range(len(sizes)):
        multiplier = 1
        for j in range(i + 1, len(sizes)):
            multiplier = multiplier * sizes[j]
        multipliers.append(multiplier)

    result = []
    for i in range(multipliers[0] * sizes[0]):
        lst = []
        num = i
        for multiplier in multipliers:
            lst.append(num // multiplier)
            num = num % multiplier
        result.append(lst)

    return result

