import itertools
import json
import logging
import os.path

import torch
from colorama import Fore

from plugins.plugins import BasePlugin
from train import EveryDreamTrainingState
from utils.sample_generator import clean_filename
import torch.nn as nn

#pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0%2Bcu121.html
import torch_sparse

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

    def on_model_load(self, **kwargs):
        text_encoder = kwargs.get('text_encoder')
        tokenizer = kwargs.get('tokenizer')
        optimizer_config: dict = kwargs.get('optimizer_config')
        def get_token_ids(t: str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t))

        # check for correctly configured text encoder training
        num_te_layers = len(text_encoder.text_model.encoder.layers)
        if False and (optimizer_config is None or
            'text_encoder_freezing' not in optimizer_config or
            optimizer_config['text_encoder_freezing'].get('freeze_embeddings') != True or
            optimizer_config['text_encoder_freezing'].get('freeze_final_layer_norm') != True or
            optimizer_config['text_encoder_freezing'].get('unfreeze_last_n_layers', num_te_layers) > 0
        ):
            required_js_fragment = {"text_encoder_freezing": {"freeze_embeddings": True, "unfreeze_last_n_layers": 0, "freeze_final_layer_norm": True}}
            logging.error(f" * {Fore.LIGHTRED_EX}Textual Inversion plugin REQUIRES the following json fragment in your optimizer config:{Fore.RESET}")
            logging.error(f" * {Fore.LIGHTRED_EX}  {json.dumps(required_js_fragment)}{Fore.RESET}")
            raise RuntimeError("Misconfigured optimizer config")

        tokens_to_train = sorted(list(set([id for t in self.config['tokens']
                                           for id in get_token_ids(t['token'])])))
        logging.info(
            f" * Textual inversion training tokens: {tokens_to_train} -> {tokenizer.convert_ids_to_tokens(tokens_to_train)}")

        self.training_token_ids = tokens_to_train

        with torch.no_grad():
            embedding_offset_index, embedding_offset_value = _build_embedding_offset_tensor(
                shape=text_encoder.get_input_embeddings().weight.shape,
                training_indices=self.training_token_ids)

            embedding_offset_coo = torch.sparse_coo_tensor(embedding_offset_index,
                                                       embedding_offset_value,
                                                       size=text_encoder.get_input_embeddings().weight.shape
                                                            ).coalesce().to(text_encoder.device)
            self.embedding_offset_sparse = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(embedding_offset_coo)
            _, _, value = self.embedding_offset_sparse.coo()
            self.embedding_offset_sparse_param = nn.Parameter(value, requires_grad=True)
            self.embedding_offset_sparse.set_value(self.embedding_offset_sparse_param, layout='coo')

            if False:
                self.embedding_offset_index = embedding_offset_index.to(text_encoder.device)
                self.embedding_offset_value = nn.Parameter(embedding_offset_value.to(text_encoder.device))
                self.embedding_offset_value.requires_grad = True
                #self.embedding_offset_index = embedding_offset_index
                #self.embedding_offset_value = nn.Parameter(embedding_offset_value)
                #self.embedding_offset_value.requires_grad = True

        def offset_hook(module, input, output):
            input_flat = input[0].reshape(-1)
            offset_flat = self.embedding_offset_sparse.index_select(0, input_flat)
            offset = offset_flat.to_dense().reshape(output.shape)
            return output + offset


        def offset_hook_silly(module, input, output):
            # Apply the sparse offset
            #return output
            embedding_offset = torch.sparse_coo_tensor(self.embedding_offset_index,
                                                       self.embedding_offset_value,
                                                       size=module.weight.shape
                                                            ).coalesce()

            input_flat = input[0].reshape(-1)
            offset_flat = embedding_offset.index_select(0, input_flat)
            offset = offset_flat.to_dense().reshape(output.shape)
            return output + offset

        def offset_hook_torchsparse(module, input, output):
            # Apply the sparse offset
            offset = torch_sparse.spmm(self.embedding_offset_index,
                                       self.embedding_offset_value,
                                       m=output.shape[2],
                                       n=output.shape[1],
                                       matrix=torch.ones_like(output))
            return output + offset
        text_encoder.text_model.embeddings.token_embedding.register_forward_hook(offset_hook)

    def add_parameters(self, text_encoder_parameters, unet_parameters):
        #text_encoder_parameters = itertools.chain(text_encoder_parameters, [self.embedding_offset_value])
        text_encoder_parameters = itertools.chain(text_encoder_parameters, [self.embedding_offset_sparse_param])
        return text_encoder_parameters, unet_parameters


    def on_step_start(self, **kwargs):
        batch = kwargs['batch']
        tokens = batch['tokens']  # a torch.stack
        self.this_batch_tokens = torch.unique(torch.flatten(tokens)).tolist()
        if len(set(self.training_token_ids).intersection(set(self.this_batch_tokens))) > 0:
            print("* training something this step")

    def on_epoch_end(self, **kwargs):
        print("* TI plugin: non-zero entries count", torch.count_nonzero(self.embedding_offset_sparse_param))

    def on_step_end(self, **kwargs):
        if False:
            ed_state: EveryDreamTrainingState = kwargs['ed_state']

            # reset the embeddings that have been touched this step, except the ones we're training, to their original state
            with (torch.no_grad()):
                embeddings = ed_state.text_encoder.get_input_embeddings()
                embeddings_to_restore = [t for t in self.this_batch_tokens if t not in self.training_token_ids]
                for t in embeddings_to_restore:
                    embeddings.weight[t] = self.original_text_embeddings[t]
        else:
            pass

    def on_model_save(self, **kwargs):
        ed_state: EveryDreamTrainingState = kwargs['ed_state']
        embeddings = ed_state.text_encoder.get_input_embeddings()
        save_folder = kwargs['save_folder']
        for token_id, token in zip(self.training_token_ids, self.training_tokens):
            _save_embedding(token=token, embedding=embeddings.weight[token_id], save_folder=save_folder)

def _save_embedding(token, embedding, save_folder):
    dict_to_save = {token: embedding}
    token_name_safe = clean_filename(token)
    ti_folder = os.path.join(save_folder, 'textual_inversions')
    os.makedirs(ti_folder, exist_ok=True)
    save_path = os.path.join(ti_folder, token_name_safe + '.bin')
    logging.info(f"Saving textual inversion for '{token}' to {save_path}")
    torch.save(dict_to_save, save_path)

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

# copied from https://stackoverflow.com/a/73395523
def slice_torch_sparse_coo_tensor(t, slices):
    """
    params:
    -------
    t: tensor to slice
    slices: slice for each dimension

    returns:
    --------
    t[slices[0], slices[1], ..., slices[n]]
    """

    t = t.coalesce()
    assert len(args) == len(t.size())
    for i in range(len(args)):
        if type(args[i]) is not torch.Tensor:
            args[i] = torch.tensor(args[i], dtype=torch.long)

    indices = t.indices()
    values = t.values()
    for dim, slice in enumerate(args):
        invert = False
        if t.size(0) * 0.6 < len(slice):
            invert = True
            all_nodes = torch.arange(t.size(0))
            unique, counts = torch.cat([all_nodes, slice]).unique(return_counts=True)
            slice = unique[counts == 1]
        if slice.size(0) > 400:
            mask = ainb_wrapper(indices[dim], slice)
        else:
            mask = ainb(indices[dim], slice)
        if invert:
            mask = ~mask
        indices = indices[:, mask]
        values = values[mask]

    return torch.sparse_coo_tensor(indices, values, t.size()).coalesce()