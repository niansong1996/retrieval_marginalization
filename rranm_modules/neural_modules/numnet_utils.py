import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import allennlp.nn.util as allennlp_utils

from overrides import overrides
from typing import Optional, Dict, Any, Tuple
from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder


def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, value: float):
    """Use this to settle the compatibility issues for numnet"""
    return allennlp_utils.replace_masked_values(tensor, mask.bool(), value)


class NumNetTextFieldEmbedder(BasicTextFieldEmbedder):
    @overrides
    def forward(
            self, text_field_input: TextFieldTensors, num_wrapping_dims: int = 0,
            only_keep_last_layer_output: bool = True, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._token_embedders.keys() != text_field_input.keys():
            message = "Mismatched token keys: %s and %s" % (
                str(self._token_embedders.keys()),
                str(text_field_input.keys()),
            )
            raise ConfigurationError(message)

        embedded_representations = []
        for key in self._ordered_embedder_keys:
            # Note: need to use getattr here so that the pytorch voodoo
            # with submodules works with multiple GPUs.
            embedder = getattr(self, "token_embedder_{}".format(key))
            forward_params = inspect.signature(embedder.forward).parameters
            forward_params_values = {}
            missing_tensor_args = set()
            for param in forward_params.keys():
                if param in kwargs:
                    forward_params_values[param] = kwargs[param]
                else:
                    missing_tensor_args.add(param)

            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(embedder)

            tensors: Dict[str, torch.Tensor] = text_field_input[key]
            if len(tensors) == 1 and len(missing_tensor_args) == 1:
                # If there's only one tensor argument to the embedder, and we just have one tensor to
                # embed, we can just pass in that tensor, without requiring a name match.
                token_vectors = embedder(list(tensors.values())[0], **forward_params_values)
            else:
                # If there are multiple tensor arguments, we have to require matching names from the
                # TokenIndexer.  I don't think there's an easy way around that.
                token_vectors = embedder(**tensors, **forward_params_values,
                                         only_keep_last_layer_output=only_keep_last_layer_output)
            if token_vectors is not None:
                # To handle some very rare use cases, we allow the return value of the embedder to
                # be None; we just skip it in that case.
                embedded_representations.append(token_vectors)

        if len(embedded_representations) > 1:
            raise ValueError(f'In the modified version of BasicTextFieldEmbedder, we only expect one embedder key '
                             f'"tokens", but we got {self._ordered_embedder_keys}')
        else:
            return embedded_representations[0]


class NumNetTransformerEmbedder(PretrainedTransformerEmbedder):
    def __init__(
            self, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transformer_model.config.update({"output_hidden_states": True})

    def _forward_unimplemented(self, *input: Any) -> None:
        raise ValueError("_forward_unimplemented is not implemented for NumNetTransformerEmbedder")

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
            only_keep_last_layer_output: bool = True) -> torch.Tensor:
        # Some of the huggingface transformers don't support type ids at all and crash when you supply
        # them. For others, you can supply a tensor of zeros, and if you don't, they act as if you did.
        # There is no practical difference to the caller, so here we pretend that one case is the same
        # as another case.
        if type_ids is not None:
            max_type_id = type_ids.max()
            if max_type_id == 0:
                type_ids = None
            else:
                if max_type_id >= self._number_of_token_type_embeddings():
                    raise ValueError("Found type ids too large for the chosen transformer model.")
                assert token_ids.shape == type_ids.shape

        transformer_mask = segment_concat_mask if self._max_length is not None else mask
        # Shape: [batch_size, num_wordpieces, embedding_size],
        # or if self._max_length is not None:
        # [batch_size * num_segments, self._max_length, embedding_size]

        # We call this with kwargs because some of the huggingface models don't have the
        # token_type_ids parameter and fail even when it's given as None.
        # Also, as of transformers v2.5.1, they are taking FloatTensor masks.
        parameters = {"input_ids": token_ids, "attention_mask": transformer_mask.float()}
        if type_ids is not None:
            parameters["token_type_ids"] = type_ids

        # THE ONLY THING CHANGED IS HERE: output all layer's hidden states
        transformer_output = self.transformer_model(**parameters)
        embeddings = transformer_output

        if only_keep_last_layer_output:
            return embeddings[0]
        else:
            return embeddings


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)


class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)


class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)


class GCN(nn.Module):

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(GCN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._dd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dq_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)

        self._dd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dq_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)

    def forward(self, d_node, q_node, d_node_mask, q_node_mask, graph, extra_factor=None):

        d_node_len = d_node.size(1)
        q_node_len = q_node.size(1)

        diagmat = torch.diagflat(torch.ones(d_node.size(1), dtype=torch.long, device=d_node.device))
        diagmat = diagmat.unsqueeze(0).expand(d_node.size(0), -1, -1)
        dd_graph = d_node_mask.unsqueeze(1) * d_node_mask.unsqueeze(-1) * (1 - diagmat)
        dd_graph_left = dd_graph * graph[:, :d_node_len, :d_node_len]
        dd_graph_right = dd_graph * (1 - graph[:, :d_node_len, :d_node_len])

        diagmat = torch.diagflat(torch.ones(q_node.size(1), dtype=torch.long, device=q_node.device))
        diagmat = diagmat.unsqueeze(0).expand(q_node.size(0), -1, -1)
        qq_graph = q_node_mask.unsqueeze(1) * q_node_mask.unsqueeze(-1) * (1 - diagmat)
        qq_graph_left = qq_graph * graph[:, d_node_len:, d_node_len:]
        qq_graph_right = qq_graph * (1 - graph[:, d_node_len:, d_node_len:])

        dq_graph = d_node_mask.unsqueeze(-1) * q_node_mask.unsqueeze(1)
        dq_graph_left = dq_graph * graph[:, :d_node_len, d_node_len:]
        dq_graph_right = dq_graph * (1 - graph[:, :d_node_len, d_node_len:])

        qd_graph = q_node_mask.unsqueeze(-1) * d_node_mask.unsqueeze(1)
        qd_graph_left = qd_graph * graph[:, d_node_len:, :d_node_len]
        qd_graph_right = qd_graph * (1 - graph[:, d_node_len:, :d_node_len])


        d_node_neighbor_num = dd_graph_left.sum(-1) + dd_graph_right.sum(-1) + dq_graph_left.sum(-1) + dq_graph_right.sum(-1)
        d_node_neighbor_num_mask = (d_node_neighbor_num >= 1).long()
        d_node_neighbor_num = replace_masked_values(d_node_neighbor_num.float(), d_node_neighbor_num_mask.bool(), 1)

        q_node_neighbor_num = qq_graph_left.sum(-1) + qq_graph_right.sum(-1) + qd_graph_left.sum(-1) + qd_graph_right.sum(-1)
        q_node_neighbor_num_mask = (q_node_neighbor_num >= 1).long()
        q_node_neighbor_num = replace_masked_values(q_node_neighbor_num.float(), q_node_neighbor_num_mask.bool(), 1)


        all_d_weight, all_q_weight = [], []
        for step in range(self.iteration_steps):
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(d_node)).squeeze(-1)
                q_node_weight = torch.sigmoid(self._node_weight_fc(q_node)).squeeze(-1)
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((d_node, extra_factor), dim=-1))).squeeze(-1)
                q_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((q_node, extra_factor), dim=-1))).squeeze(-1)

            all_d_weight.append(d_node_weight)
            all_q_weight.append(q_node_weight)

            self_d_node_info = self._self_node_fc(d_node)
            self_q_node_info = self._self_node_fc(q_node)

            dd_node_info_left = self._dd_node_fc_left(d_node)
            qd_node_info_left = self._qd_node_fc_left(d_node)
            qq_node_info_left = self._qq_node_fc_left(q_node)
            dq_node_info_left = self._dq_node_fc_left(q_node)

            dd_node_weight = replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                dd_graph_left,
                0)

            qd_node_weight = replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                qd_graph_left,
                0)

            qq_node_weight = replace_masked_values(
                q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                qq_graph_left,
                0)

            dq_node_weight = replace_masked_values(
                q_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                dq_graph_left,
                0)

            dd_node_info_left = torch.matmul(dd_node_weight, dd_node_info_left)
            qd_node_info_left = torch.matmul(qd_node_weight, qd_node_info_left)
            qq_node_info_left = torch.matmul(qq_node_weight, qq_node_info_left)
            dq_node_info_left = torch.matmul(dq_node_weight, dq_node_info_left)


            dd_node_info_right = self._dd_node_fc_right(d_node)
            qd_node_info_right = self._qd_node_fc_right(d_node)
            qq_node_info_right = self._qq_node_fc_right(q_node)
            dq_node_info_right = self._dq_node_fc_right(q_node)

            dd_node_weight = replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                dd_graph_right,
                0)

            qd_node_weight = replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                qd_graph_right,
                0)

            qq_node_weight = replace_masked_values(
                q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                qq_graph_right,
                0)

            dq_node_weight = replace_masked_values(
                q_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                dq_graph_right,
                0)

            dd_node_info_right = torch.matmul(dd_node_weight, dd_node_info_right)
            qd_node_info_right = torch.matmul(qd_node_weight, qd_node_info_right)
            qq_node_info_right = torch.matmul(qq_node_weight, qq_node_info_right)
            dq_node_info_right = torch.matmul(dq_node_weight, dq_node_info_right)


            agg_d_node_info = (dd_node_info_left + dd_node_info_right + dq_node_info_left + dq_node_info_right) / d_node_neighbor_num.unsqueeze(-1)
            agg_q_node_info = (qq_node_info_left + qq_node_info_right + qd_node_info_left + qd_node_info_right) / q_node_neighbor_num.unsqueeze(-1)

            d_node = F.relu(self_d_node_info + agg_d_node_info)
            q_node = F.relu(self_q_node_info + agg_q_node_info)


        all_d_weight = [weight.unsqueeze(1) for weight in all_d_weight]
        all_q_weight = [weight.unsqueeze(1) for weight in all_q_weight]

        all_d_weight = torch.cat(all_d_weight, dim=1)
        all_q_weight = torch.cat(all_q_weight, dim=1)

        return d_node, q_node, all_d_weight, all_q_weight # d_node_weight, q_node_weight


def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
    """
    This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
    in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
    directly import this function without the class.

    We call the inputs "logits" - they could either be unnormalized logits or normalized log
    probabilities.  A log_softmax operation is a constant shifting of the entire logit
    vector, so taking an argmax over either one gives the same result.
    """
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length),
                                          device=device)).log()
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    # best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
    _, best_spans = valid_span_log_probs.view(batch_size, -1).topk(20, dim=-1)

    # (batch_size, 20)
    span_start_indices = best_spans // passage_length
    span_end_indices = best_spans % passage_length

    # (batch_size, 20, 2)
    return torch.stack([span_start_indices, span_end_indices], dim=-1)

def best_answers_extraction(best_spans, span_num, original_str, offsets, offset_start):
    predicted_span = tuple(best_spans.detach().cpu().numpy())
    predict_answers = []
    predict_offsets = []
    for i in range(20):
        start_offset = offsets[predicted_span[i][0] - offset_start][0] if predicted_span[i][0] - offset_start < len(offsets) else offsets[-1][0]
        end_offset = offsets[predicted_span[i][1] - offset_start][1] if predicted_span[i][1] - offset_start < len(offsets) else offsets[-1][1]
        predict_answer = original_str[start_offset:end_offset]
        predict_offset = (start_offset, end_offset)
        if len(predict_answers) == 0 or all([len(set(item.split()) & set(predict_answer.split())) == 0 for item in predict_answers]):
            predict_answers.append( predict_answer)
            predict_offsets.append( predict_offset)
        if len(predict_answers) >= span_num:
            break
    return predict_answers, predict_offsets

def convert_number_to_str(number):
    if isinstance(number, int):
        return str(number)

    # we leave at most 3 decimal places
    num_str = '%.3f' % number

    for i in range(3):
        if num_str[-1] == '0':
            num_str = num_str[:-1]
        else:
            break

    if num_str[-1] == '.':
        num_str = num_str[:-1]

    # if number < 1, them we will omit the zero digit of the integer part
    if num_str[0] == '0' and len(num_str) > 1:
        num_str = num_str[1:]

    return num_str
