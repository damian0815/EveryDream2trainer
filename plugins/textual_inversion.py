import itertools
import json
import logging
import os.path
import random
from typing import Optional, cast

import torch
from colorama import Fore
from transformers import CLIPTextModel, CLIPTokenizer

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
        self.training_words: list[str] = []
        self.training_words_to_tokens: dict[str, list[int]] = {}
        self.embedding_offsets_individual: nn.Parameter = None
        self.text_encoder: CLIPTextModel = None
        self.tokenizer: CLIPTokenizer = None
        self.log_folder: str = None

    @property
    def fallback_word(self):
        fallback_word = self.config.get('fallback_word', self.training_words[0])
        tokens = self.training_words_to_tokens[fallback_word]
        return self.tokenizer.decode(tokens)



    def on_model_load(self, **kwargs):
        self.text_encoder = kwargs.get('text_encoder')
        self.tokenizer = kwargs.get('tokenizer')
        optimizer_config: dict = kwargs.get('optimizer_config')
        def get_token_ids(t: str):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(t))

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

        if 'lerp' in self.config:
            lerp_target = self.config['lerp']['lerp_target']
            from safetensors.torch import load_file
            target_te_data = load_file(lerp_target)
            key = 'text_model.embeddings.token_embedding.weight'
            self.lerp_base_weights = self.text_encoder.get_input_embeddings().weight
            print('fetcged lerp_target_weights with shape', self.lerp_base_weights.shape)
            self.lerp_target_weights = target_te_data[key].to(self.lerp_base_weights.device)
            print('loaded lerp_target_weights with shape', self.lerp_target_weights.shape)
            self.lerp_target_steps = self.config['lerp']['steps']
            self.lerp_current_step = 0
        else:
            self.lerp_target_weights = None

            #text_model: CLIPTextModel = self.text_encoder.text_model
            self.training_words = []
            for token_config in self.config['tokens']:
                # {"token": "*", "initializer": "corpse, abstract, group", "vector_length": 8}
                training_word = token_config['token']

                self.training_words.append(training_word)
                if token_config.get('overwrite_existing', False):
                    self.training_words_to_tokens[training_word] = get_token_ids(training_word)
                else:
                    vector_length = token_config.get('vector_length', 1)
                    tokens_to_add = self.expand_tokens(training_word, vector_length)
                    num_tokens_added = self.tokenizer.add_tokens(tokens_to_add)
                    self.text_encoder.resize_token_embeddings(len(self.text_encoder.get_input_embeddings().weight) + num_tokens_added)
                    self.training_words_to_tokens[training_word] = [tid
                                                                    for w in tokens_to_add
                                                                    for tid in get_token_ids(w)]

            print("after adding/updating tokens: input_embeddings has length", len(self.text_encoder.get_input_embeddings().weight))

            tokens_to_train = sorted(list(set([tid
                                               for tids in self.training_words_to_tokens.values()
                                               for tid in tids])))
            logging.info(
                f" * Text embedding unlocked tokens: {tokens_to_train} -> {self.tokenizer.convert_ids_to_tokens(tokens_to_train)}")
            self.training_token_ids = torch.tensor(tokens_to_train, device=self.text_encoder.device, dtype=torch.int64)

            embeddings: nn.Embedding = self.text_encoder.get_input_embeddings()
            with torch.no_grad():
                stds = embeddings.weight.std(dim=0)
                means = embeddings.weight.mean(dim=0)
                for t in self.config['tokens']:
                    tids_to_initialize = self.training_words_to_tokens[t['token']]
                    # always calculate random weights, even if they're not used, to ensure persistent
                    # behaviour with same seed
                    random_weights = {tid: torch.normal(mean=means, std=stds)
                                      for tid in tids_to_initialize}
                    if t.get('initialize_random', False):
                        print(f'initializing {tids_to_initialize} randomly')
                        for tid in tids_to_initialize:
                            embeddings.weight.data[tid] = random_weights[tid]
                    elif 'initializer' in t:
                        initializer = t['initializer']
                        print(f'initializing {tids_to_initialize} from {initializer}')
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

    def on_model_save(self, **kwargs):
        # ed_state = kwargs['ed_state']
        save_path = kwargs['save_path']
        te_path = os.path.join(save_path, 'text_embeddings')
        self.save(te_path)

    def save(self, save_path, suffix: str=''):
        self.bounce_down_weights()
        with torch.no_grad():
            os.makedirs(save_path, exist_ok=True)
            embeddings: nn.Embedding = self.text_encoder.get_input_embeddings()
            offset_weights = _apply_weight_offsets(self.training_token_ids, embeddings, self.embedding_offsets_individual)
            for word, tokens in self.training_words_to_tokens.items():
                word: str
                tokens: list[int]
                embeddings_stack = offset_weights[tokens]

                dict = {
                    'string_to_token': {'*': 265},
                    'string_to_param': {'*': embeddings_stack.detach().clone().cpu()},
                    'name': word,
                }
                save_path = os.path.join(save_path, f'{word}{suffix}.pt')
                torch.save(dict, save_path)


    def expand_tokens(self, base_token_text, vector_length) -> list[str]:
        return [f'{base_token_text}.{i}' for i in range(vector_length)]

    def add_parameters(self, text_encoder_parameters, unet_parameters):
        if self.lerp_target_weights is None:
            text_encoder_parameters = itertools.chain(text_encoder_parameters,
                    [('te_offset', self.embedding_offsets_individual)])
        return text_encoder_parameters, unet_parameters

    def on_step_end(self, **kwargs):
        if self.lerp_target_weights is not None:
            global_step = kwargs['global_step']
            lerp_alpha = global_step / self.lerp_target_steps
            if lerp_alpha > 1:
                return
            new_weights = self.lerp_target_weights * lerp_alpha + self.lerp_base_weights * (1-lerp_alpha)
            with torch.no_grad():
                self.text_encoder.get_input_embeddings().weight.data = new_weights.data.to(dtype=self.text_encoder.get_input_embeddings().weight.dtype)


    def on_epoch_end(self, **kwargs):
        if self.lerp_target_weights is None:
            if torch.count_nonzero(self.embedding_offsets_individual).item() == 0:
                logging.warning(" * TextualInversionPlugin: warning: nothing has happened (possible misconfiguration? check batch size)")
            self.bounce_down_weights()

        te_path = os.path.join(self.log_folder, 'text_embeddings')
        epoch = kwargs['epoch']
        self.save(te_path, suffix=f'-ep{epoch:03}')

    def on_training_start(self, **kwargs):
        self.log_folder = kwargs['log_folder']

    def on_training_end(self, **kwargs):
        self.bounce_down_weights()


    def bounce_down_weights(self):
        #print("NOT doing bounce_down_weights")
        with torch.no_grad():
            # bounce offsets down into actual embeddings array and reset offsets
            embeddings = self.text_encoder.get_input_embeddings()
            offset_weights = _apply_weight_offsets(self.training_token_ids,
                                                   original_embeddings=embeddings, # type: ignore
                                                   embedding_offsets_individual=self.embedding_offsets_individual)
            embeddings.weight.data = offset_weights.data
            self.embedding_offsets_individual.data = torch.zeros_like(self.embedding_offsets_individual.data)

    def transform_caption(self, caption:str):
        if len(caption.strip()) == 0:
            imagenet_caption = self.make_imagenet_caption()
            #print('made caption:', imagenet_caption)
            return imagenet_caption

        if (self.fallback_word is not None
                and all(re.search('(^|[\W])'+word+'([\W]|$)', caption) is None
                    for word in self.training_words)
        ):
            print(f"caption '{caption}' is missing text training terms - inserting '{self.fallback_word}' to prevent NaN")
            return self.fallback_word + ' ' + caption
        else:
            return caption


    def make_imagenet_caption(self):
        imagenet_templates_small = [
            "a photo of a {}",
            "a rendering of a {}",
            "a cropped photo of the {}",
            "the photo of a {}",
            "a photo of a clean {}",
            "a photo of a dirty {}",
            "a dark photo of the {}",
            "a photo of my {}",
            "a photo of the cool {}",
            "a close-up photo of a {}",
            "a bright photo of the {}",
            "a cropped photo of a {}",
            "a photo of the {}",
            "a good photo of the {}",
            "a photo of one {}",
            "a close-up photo of the {}",
            "a rendition of the {}",
            "a photo of the clean {}",
            "a rendition of a {}",
            "a photo of a nice {}",
            "a good photo of a {}",
            "a photo of the nice {}",
            "a photo of the small {}",
            "a photo of the weird {}",
            "a photo of the large {}",
            "a photo of a cool {}",
            "a photo of a small {}",
        ]
        return random.choice(imagenet_templates_small).replace('{}', self.fallback_word)



def _embedding_forward_individual_hook(module, input_args, output):
    offset_weight = _apply_weight_offsets(module.training_token_ids, module, module.embedding_offsets_individual)
    # offset_weight = self.apply_weight_offsets()
    # re-implement stock nn.Embedding forward()
    return F.embedding(
        input_args[0], offset_weight, module.padding_idx, module.max_norm,
        module.norm_type, module.scale_grad_by_freq, module.sparse)

def _apply_weight_offsets(
        training_token_ids: torch.Tensor,
        original_embeddings: nn.Embedding,
        embedding_offsets_individual: torch.Tensor
) -> torch.Tensor:
    index = training_token_ids
    offset_weight = original_embeddings.weight.index_add(
        0, index, embedding_offsets_individual
    )
    return offset_weight
