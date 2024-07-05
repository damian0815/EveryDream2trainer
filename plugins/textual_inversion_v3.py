import itertools
import json
import logging
import os.path
from typing import Optional, cast

import torch
from colorama import Fore
from transformers import CLIPTextModel

from plugins.plugins import BasePlugin
from train import EveryDreamTrainingState
from utils.sample_generator import clean_filename
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
        "freeze_embeddings": false,
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

        """
        with torch.no_grad():
            embedding_offset_index, embedding_offset_value = _build_embedding_offset_tensor(
                shape=text_encoder.get_input_embeddings().weight.shape,
                training_indices=self.training_token_ids)

            self.embedding_offset_index = embedding_offset_index.to(text_encoder.device)
            self.embedding_offset_value = nn.Parameter(embedding_offset_value.to(text_encoder.device))
            self.embedding_offset_value.requires_grad = True
            text_encoder.get_input_embeddings().weight.requires_grad = False
        """

        embedding_len = self.text_encoder.get_input_embeddings().weight.shape[1]
        embedding_offsets_individual = [
            torch.zeros([embedding_len],
                                     dtype=torch.float32,
                                     device=self.text_encoder.device,
                                     requires_grad=True)
                                             for _ in self.training_token_ids]
        self.embedding_offsets_individual = nn.Parameter(torch.stack(embedding_offsets_individual), requires_grad=True)

        #torch.autograd.set_detect_anomaly(True)

        embeddings.training_token_ids = self.training_token_ids
        embeddings.embedding_offsets_individual = self.embedding_offsets_individual
        embeddings.register_forward_hook(_embedding_forward_individual_hook)

        # self.text_encoder.text_model.embeddings is a CLIPTextEmbedding,
        # but get_input_embeddings() returns an nn.Embedding (== self.text_encoder.text_model.token_embedding)
        #embeddings: nn.Embedding = self.text_encoder.get_input_embeddings()
        #embeddings.forward = embedding_forward_individual.__get__(embeddings, nn.Embedding)
        #forward_func_type = type(embeddings.forward)
        #embeddings.forward = forward_func_type(embedding_forward_individual, embeddings, nn.Embedding)

        def _backward_hook_test(module, grad_input, grad_output):
            print(f"backward hook on {type(module)} with grad_output {grad_output}")
            print(f"self grad:",
                  module.weight.requires_grad,
                  module.weight.grad,
                  module.weight.grad_fn)
            print(f"self.embedding_offsets_individual:",
                  module.embedding_offsets_individual.requires_grad,
                  module.embedding_offsets_individual.grad,
                  module.embedding_offsets_individual.grad_fn)

            #print([type(x) for x in grad_output])
            #print(len([x for x in grad_output if hasattr(x, "requires_grad") and x.requires_grad]))
            return grad_input

        #embeddings.register_full_backward_hook(_backward_hook_test)
        #embeddings.weight.requires_grad = True

        #bound_method = embedding_forward_individual.__get__(embeddings, embeddings.__class__)
        #setattr(embeddings, 'forward', bound_method)


    def add_parameters(self, text_encoder_parameters, unet_parameters):
        #text_encoder_parameters = itertools.chain(text_encoder_parameters, [self.embedding_offset_value])
        text_encoder_parameters = itertools.chain(text_encoder_parameters, [self.embedding_offsets_individual])
        #text_encoder_parameters = itertools.chain(text_encoder_parameters, [self.embedding_offset_sparse_param])
        return text_encoder_parameters, unet_parameters


    def on_epoch_end(self, **kwargs):
        if torch.count_nonzero(self.embedding_offsets_individual).item() == 0:
            logging.warning(" * TextualInversionPlugin: warning: nothing has happened (possible misconfiguration?)")
        # bounce offsets down into actual embeddings array and reset offsets
        with torch.no_grad():
            embeddings = self.text_encoder.get_input_embeddings()
            offset_weights = _apply_weight_offsets(self.training_token_ids,
                                                   original_embeddings=embeddings, # type: ignore
                                                   embedding_offsets_individual=self.embedding_offsets_individual)
            embeddings.weight.data = offset_weights.data
            self.embedding_offsets_individual.zero_()

    """def on_model_save(self, **kwargs):
        ed_state: EveryDreamTrainingState = kwargs['ed_state']
        embeddings = ed_state.text_encoder.get_input_embeddings()
        save_folder = kwargs['save_folder']
        for token_id, token in zip(self.training_token_ids, self.training_token_ids):
            _save_embedding(token=token, embedding=embeddings.weight[token_id], save_folder=save_folder)
    """

    def transform_caption(self, caption:str):
        if all(re.search('(^|[\W])'+word+'([\W]|$)', caption) is None for word in self.training_words):
            print(f"caption '{caption}' is missing TI training terms - inserting '{self.fallback_word}'")
            return self.fallback_word + ' ' + caption
        else:
            return caption

def _save_embedding(token, embedding, save_folder):
    dict_to_save = {token: embedding}
    token_name_safe = clean_filename(token)
    ti_folder = os.path.join(save_folder, 'textual_inversions')
    os.makedirs(ti_folder, exist_ok=True)
    save_path = os.path.join(ti_folder, token_name_safe + '.bin')
    logging.info(f"Saving textual inversion for '{token}' to {save_path}")
    torch.save(dict_to_save, save_path)


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


def _build_embedding_offset_tensor(shape: tuple, training_indices: list):
    # Define indices and values for the sparse matrix
    # indices work like this:
    # for sparse entries [a,1], [a,2], [a,3], [b,1], [b,2], [b,3]
    #  -> the indices tensor should look like this: [[a, a, a, b, b, b], [1, 2, 3, 1, 2, 3]]
    # [a, a, a, b, b, b]
    i_indices = torch.repeat_interleave(torch.tensor(training_indices), repeats=shape[1])
    # [1, 2, 3, 1, 2, 3]
    j_indices = torch.arange(shape[1]).repeat(len(training_indices))
    # combine
    indices = torch.stack([i_indices, j_indices])
    # zero offsets as default
    values = torch.zeros(i_indices.shape, dtype=torch.float32)

    # Create the sparse embedding offset
    #look into using pytorch_sparse here c
    embedding_offset_index = indices
    embedding_offset_value = values
    return embedding_offset_index, embedding_offset_value
