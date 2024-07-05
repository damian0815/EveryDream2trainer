import itertools
import json
import logging
import os.path
from typing import Optional, cast

import torch
from colorama import Fore
from transformers import CLIPTextModel

from plugins.plugins import BasePlugin
import torch.nn as nn
import torch.nn.functional as F
import re

""" 
This plugin adds custom tokens to the tokenizer and trains just these tokens, with the rest of the text encoder
disabled/frozen.

token/initialization config is in textual_inversion.json, same folder as this .py file.

For pure Textual Inversion training:
  "disable_textenc_training": false,
  "disable_unet_training": true
(Or you could unet training on too if you want, I didn't test this.)
  

In optimizer.json, the following "text_encoder_freezing" section is *required*:
    "text_encoder_freezing": {
        "unfreeze_last_n_layers": 0,
        "freeze_embeddings": true,
        "freeze_final_layer_norm": true
    }
In addition, you'll need a very high LR on the TE - maybe even as high as 1e-3. I recommend using the LR finder method.

"""

class TextualInversionPlugin(BasePlugin):

    def __init__(self):
        path = os.path.join(os.path.dirname(__file__), "textual_inversion.json")
        logging.info(f" * Textual Inversion plugin instantiated, loading config from {path}")
        with open(path, 'rt') as f:
            self.config = json.load(f)
        self.this_batch_tokens = None
        self.training_token_ids = None
        self.original_text_embeddings = None
        self.training_words = []
        self.embedding_offsets_individual = None
        self.text_encoder: Optional[CLIPTextModel] = None

    @property
    def fallback_word(self):
        return self.config['fallback_word']

    def on_model_load(self, **kwargs):
        self.text_encoder = kwargs.get('text_encoder')
        tokenizer = kwargs.get('tokenizer')
        optimizer_config: dict = kwargs.get('optimizer_config')
        def get_token_ids(t: str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t))

        # check for correctly configured text encoder training
        num_te_layers = len(self.text_encoder.text_model.encoder.layers)
        if (optimizer_config is None or
            'text_encoder_freezing' not in optimizer_config or
            optimizer_config['text_encoder_freezing'].get('freeze_embeddings') != True or
            optimizer_config['text_encoder_freezing'].get('freeze_final_layer_norm') != True or
            optimizer_config['text_encoder_freezing'].get('unfreeze_last_n_layers', num_te_layers) > 0
        ):
            required_js_fragment = {"text_encoder_freezing": {"freeze_embeddings": True, "unfreeze_last_n_layers": 0, "freeze_final_layer_norm": True}}
            logging.error(f" * {Fore.LIGHTRED_EX}Textual Inversion plugin REQUIRES the following json fragment in your optimizer config:{Fore.RESET}")
            logging.error(f" * {Fore.LIGHTRED_EX}  {json.dumps(required_js_fragment)}{Fore.RESET}")
            logging.error(f" * {Fore.LIGHTRED_EX}You have:{Fore.RESET}")
            logging.error(f" * {Fore.LIGHTRED_EX}  " + json.dumps(optimizer_config.get("text_encoder_freezing", {})) + f"{Fore.RESET}")

            raise RuntimeError("Misconfigured optimizer config")

        self.training_words = [t['token'] for t in self.config['tokens']]
        tokens_to_train = sorted(list(set([tid
                                           for w in self.training_words
                                           for tid in get_token_ids(w)])))
        logging.info(
            f" * Text embedding unlocked tokens: {tokens_to_train} -> {tokenizer.convert_ids_to_tokens(tokens_to_train)}")
        self.training_token_ids = torch.tensor(tokens_to_train, device=self.text_encoder.device, dtype=torch.int64)

        embeddings: nn.Embedding = self.text_encoder.get_input_embeddings()
        with torch.no_grad():
            stds = embeddings.weight.std(dim=0)
            means = embeddings.weight.mean(dim=0)
            for t in self.config['tokens']:
                tids_to_initialize = get_token_ids(t['token'])
                # always calculate random weights, even if they're not used, to ensure persistent
                # behaviour with same seed
                random_weights = {tid: torch.normal(mean=means, std=stds)
                                  for tid in tids_to_initialize}
                if t.get('initialize_random', False):
                    for tid in tids_to_initialize:
                        embeddings.weight.data[tid] = random_weights[tid]
                elif 'initializer' in t:
                    initializer = t['initializer']
                    initializer_tids = get_token_ids(initializer)
                    initializer_weights = [embeddings.weight[i] for i in initializer_tids]
                    for i, t in enumerate(tids_to_initialize):
                        embeddings.weight.data[t] = initializer_weights[i % len(initializer_weights)]

        embedding_len = self.text_encoder.get_input_embeddings().weight.shape[1]
        embedding_offsets_individual = [
            torch.zeros([embedding_len],
                                     dtype=torch.float32,
                                     device=self.text_encoder.device,
                                     requires_grad=True)
                                             for _ in self.training_token_ids]
        self.embedding_offsets_individual = nn.Parameter(torch.stack(embedding_offsets_individual), requires_grad=True)

        embeddings.training_token_ids = self.training_token_ids
        embeddings.embedding_offsets_individual = self.embedding_offsets_individual
        embeddings.register_forward_hook(_embedding_forward_individual_hook)


    def add_parameters(self, text_encoder_parameters, unet_parameters):
        text_encoder_parameters = itertools.chain(text_encoder_parameters,
                                                  [self.embedding_offsets_individual])
        return text_encoder_parameters, unet_parameters


    def on_epoch_end(self, **kwargs):
        if torch.count_nonzero(self.embedding_offsets_individual).item() == 0:
            logging.warning(" * TextualInversionPlugin: warning: nothing has happened (possible misconfiguration?)")
        with torch.no_grad():
            # bounce offsets down into actual embeddings array and reset offsets
            embeddings = self.text_encoder.get_input_embeddings()
            offset_weights = _apply_weight_offsets(self.training_token_ids,
                                                   original_embeddings=embeddings, # type: ignore
                                                   embedding_offsets_individual=self.embedding_offsets_individual)
            embeddings.weight.data = offset_weights.data
            self.embedding_offsets_individual.zero_()

    def transform_caption(self, caption:str):
        if all(re.search('(^|[\W])'+word+'([\W]|$)', caption) is None for word in self.training_words):
            print(f"caption '{caption}' is missing text training terms - inserting '{self.fallback_word}' to prevent NaN")
            return self.fallback_word + ' ' + caption
        else:
            return caption

def _embedding_forward_individual_hook(module, input_args, output):
    offset_weight = _apply_weight_offsets(module.training_token_ids, module, module.embedding_offsets_individual)
    # offset_weight = self.apply_weight_offsets()
    # re-implement stock nn.Embedding forward()
    return F.embedding(
        input_args[0], offset_weight, module.padding_idx, module.max_norm,
        module.norm_type, module.scale_grad_by_freq, module.sparse)

def _apply_weight_offsets(training_token_ids: torch.Tensor, original_embeddings: nn.Embedding, embedding_offsets_individual: torch.Tensor):
    index = training_token_ids
    offset_weight = original_embeddings.weight.index_add(
        0, index, embedding_offsets_individual
    )
    return offset_weight
