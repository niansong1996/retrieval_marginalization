import torch

from typing import List, Optional
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.nn.util import replace_masked_values, min_value_of_dtype


def add_special_tokens_to_sent_pair(transformer_tokenizer: PretrainedTransformerTokenizer,
                                    tokens1: List[Token], tokens2: List[Token]) -> List[Token]:
    def with_new_type_id(tokens: List[Token], type_id: int) -> List[Token]:
        return [dataclasses.replace(t, type_id=type_id) for t in tokens]

    # We add special tokens and also set token type ids.
    import dataclasses

    return (
            transformer_tokenizer.sequence_pair_start_tokens
            + with_new_type_id(tokens1, transformer_tokenizer.sequence_pair_first_token_type_id)
            + transformer_tokenizer.sequence_pair_mid_tokens
            + with_new_type_id(tokens2, transformer_tokenizer.sequence_pair_second_token_type_id)
            + transformer_tokenizer.sequence_pair_end_tokens
    )


def replace_masked_values_with_big_negative_number(x: torch.Tensor, mask: torch.Tensor):
    """
    Replace the masked values in a tensor something really negative so that they won't
    affect a max operation.
    """
    return replace_masked_values(x, mask, min_value_of_dtype(x.dtype))

