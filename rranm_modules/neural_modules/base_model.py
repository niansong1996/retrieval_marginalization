from typing import Any, Dict, List, Optional

import torch
import logging

from torch import nn

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder

from rranm_modules.utils.iirc_metric import SetFMeasure


class TransformerBasedModel(Model):
    def __init__(self, vocab: Vocabulary, loaded_transformer_embedder: BasicTextFieldEmbedder = None,
                 transformer_model_name: str = "bert-base-cased", print_trajectory: bool = False, **kwargs):
        super().__init__(vocab, **kwargs)

        if loaded_transformer_embedder:
            self._text_field_embedder = loaded_transformer_embedder
        else:
            self._text_field_embedder = BasicTextFieldEmbedder(
                {"tokens": PretrainedTransformerEmbedder(transformer_model_name)})

        self.print_trajectory = print_trajectory

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        raise ValueError('TransformerBasedModel is an abstract class and shouldn\'t be initialized!')

    def log_trajectory(self, info: str):
        if self.print_trajectory:
            logging.info(info)

