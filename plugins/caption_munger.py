import json
import logging
import random
import threading
import time

from transformers import CLIPTokenizer

from plugins.plugins import BasePlugin

"""
shuffle_sentences_p = 0.8
shuffle_phrases_within_sentences_p = 0.3
keep_first_phrase_p = 0.6
keep_first_sentence_p = 0.9

truncate_sentences_p = 0.5
ending_dot_p = 0.5
replace_dots_with_commas_p = 0.3
"""

shuffle_sentences_p = 0.02
too_long_caption_shuffle_sentences_p = 0.5
shuffle_phrases_within_sentences_p = 0.01
keep_first_phrase_p = 1
keep_first_sentence_p = 0.98

truncate_sentences_p = 0
ending_dot_p = 0.5
replace_dots_with_commas_p = 0.02

class CaptionMungerPlugin(BasePlugin):

    tokenizer: CLIPTokenizer|None = None

    def on_model_load(self, **kwargs):
        self.tokenizer = kwargs['tokenizer']

    def transform_caption_parts(self, caption:str) -> str:

        parts = [s.strip() for s in caption.split("||")]

        shuffle_p = 0.3
        truncate_p = 0.3
        if shuffle_p > random.random():
            random.shuffle(parts)
        if truncate_p > random.random():
            truncate_after_idx = random.randint(0, len(parts))
            parts = parts[:truncate_after_idx]
        return ", ".join(parts)


    def transform_caption_json_raw(self, captions_json) -> dict[str,str]:
        try:
            d = json.loads(captions_json)
        except Exception as e:
            logging.error(f"unable to load json from {captions_json}: {e}")
            raise
        transformed_json = {k.lower(): (self.transform_caption(v)
                                          if (v and len(v.strip())>0)
                                          else None)
                for k, v in d.items()}
        #print(f"transformed caption from '{d}' to '{transformed_json}'")
        return transformed_json

    def transform_caption(self, caption_in:str) -> str | dict[str, str]:
        caption = caption_in
        if caption.startswith("<<json>>"):
            return self.transform_caption_json_raw(
                caption.replace("<<json>>", "")
            )

        prefix = ""
        suffix = ""
        if "<<then>>" in caption:
            parts = [p.strip() for p in caption.split("<<then>>")]
            prefix = handle_shufbreak(parts[0],
                                      keep_first_sentence_p=1,
                                      shuffle_sentences_p=shuffle_sentences_p,
                                      truncate_sentences_p=0)
            caption = "<<shufbreak>>".join(parts[1:])

        if "<<finally>>" in caption:
            parts = [p.strip() for p in caption.split("<<finally>>")]
            suffix = handle_shufbreak(parts[-1],
                                      keep_first_sentence_p=0,
                                      shuffle_sentences_p=shuffle_sentences_p,
                                      truncate_sentences_p=truncate_sentences_p)
            caption = "<<shufbreak>>".join(parts[:-1])

        tokens = _get_tokens_full(caption, prefix, suffix, self.tokenizer)
        if len(tokens) > 75:
            actual_shuffle_sentences_p = too_long_caption_shuffle_sentences_p
        else:
            actual_shuffle_sentences_p = shuffle_sentences_p

        out_caption = handle_shufbreak(
            caption,
            keep_first_sentence_p=0 if len(prefix.strip()) > 0 else keep_first_sentence_p,
            shuffle_sentences_p=actual_shuffle_sentences_p,
            truncate_sentences_p=truncate_sentences_p,
        )

        if replace_dots_with_commas_p > random.random():
            out_caption = out_caption.replace(". ", ", ")

        if len(prefix.strip()) > 0:
            out_caption = prefix + '. ' + out_caption

        if len(suffix.strip()) > 0:
            out_caption = out_caption.strip()
            if out_caption[-1] != '.':
                out_caption += '. '
            out_caption += suffix

        if ending_dot_p > random.random():
            out_caption += "."

        #print(f"transformed caption from '{caption}' to '{out_caption}'")
        if 'cleavage<' in out_caption or '<cleavage' in out_caption:
            print("broke!")
            time.sleep(1)
            return self.transform_caption(caption_in)
        return out_caption

def _get_tokens_full(caption_with_shufbreaks, prefix, suffix, tokenizer: CLIPTokenizer) -> list[str]:
    full_caption = ''

    if len(prefix.strip()) > 0:
        full_caption += prefix
    full_caption += caption_with_shufbreaks.replace('<<shufbreak>>', '. ')
    if len(suffix.strip()) > 0:
        full_caption += suffix

    return tokenizer.tokenize(full_caption)


def partial_shuffle(l, factor=5):
    n = len(l)
    if n == 0:
        return
    for _ in range(factor):
        a, b = random.randrange(n), random.randrange(n)
        l[b], l[a] = l[a], l[b]

def shuffle_on_doublebar(sentence: str) -> str:
    phrases = [s.strip() for s in sentence.split('||')]
    phrases = [p for p in phrases if len(p) > 0]
    # possibly preserve first phrase
    first_phrase = phrases.pop(0) if keep_first_phrase_p > random.random() else None
    partial_shuffle(phrases, factor=len(phrases))
    if first_phrase is not None:
        phrases.insert(0, first_phrase)
    shuffled_sentence = ", ".join(phrases)
    return shuffled_sentence

def handle_shufbreak(caption: str,
                     keep_first_sentence_p,
                     shuffle_sentences_p,
                     truncate_sentences_p) -> str:
    # split to sentences
    in_sentences = [s.strip() for s in caption.split("<<shufbreak>>")]
    # remove zero-length sentences (multiple . and trailing .)
    in_sentences = [s for s in in_sentences if len(s) > 0]
    if len(in_sentences) == 0:
        # empty string
        return caption

    out_sentences = []

    # unused if not truncating
    truncate_after_idx = random.randint(0, len(in_sentences))

    for i in range(len(in_sentences)):
        if shuffle_phrases_within_sentences_p > random.random():
            in_sentences[i] = shuffle_on_doublebar(in_sentences[i])
        else:
            in_sentences = [s.replace('||', ', ') for s in in_sentences]

    if keep_first_sentence_p > random.random():
        out_sentences.append(in_sentences[0])
        in_sentences.pop(0)
        truncate_after_idx -= 1

    if shuffle_sentences_p > random.random():
        random.shuffle(in_sentences)

    if truncate_sentences_p > random.random():
        in_sentences = in_sentences[:truncate_after_idx + 1]

    out_sentences += in_sentences

    out_caption = ". ".join(out_sentences)
    return out_caption
