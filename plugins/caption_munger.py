import random
from plugins.plugins import BasePlugin

shuffle_sentences_p = 0.9
shuffle_phrases_within_sentences_p = 0.3
keep_first_phrase_p = 1
keep_first_sentence_p = 1

truncate_sentences_p = 0.95
ending_dot_p = 0.5
replace_dots_with_commas_p = 0.3

fondle_p = 0.5

class CaptionMungerPlugin(BasePlugin):

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


    def transform_caption(self, caption:str) -> str:

        prefix = ""
        if "<<then>>" in caption:
            parts = [p.strip() for p in caption.split("<<then>>")]
            prefix = parts[0]
            caption = "<<or>>".join(parts[1:])

        if "<<or>>" in caption:
            options = [o.strip() for o in caption.split("<<or>>")]
            caption = random.choice(options)

        if "||" in caption:
            #print("ignoring || in caption")
            return self.transform_caption_parts(caption)

        # split to sentences
        in_sentences = [s.strip() for s in caption.split(".")]
        # remove zero-length sentences (multiple . and trailing .)
        in_sentences = [s for s in in_sentences if len(s)>0]
        out_sentences = []

        # unused if not truncating
        truncate_after_idx = random.randint(0, len(in_sentences))

        if shuffle_phrases_within_sentences_p > random.random():
            for i in range(len(in_sentences)):
                sentence = in_sentences[i]
                phrases = [s.strip() for s in sentence.split(',')]
                phrases = [p for p in phrases if len(p) > 0]
                # possibly preserve first phrase
                first_phrase = phrases.pop(0) if keep_first_phrase_p > random.random() else None
                partial_shuffle(phrases, factor=len(phrases))
                if first_phrase is not None:
                    phrases.insert(0, first_phrase)
                shuffled_sentence = ", ".join(phrases)
                in_sentences[i] = shuffled_sentence

        if keep_first_sentence_p > random.random():
            out_sentences.append(in_sentences[0])
            in_sentences.pop(0)
            truncate_after_idx -= 1

        if shuffle_sentences_p > random.random():
            random.shuffle(in_sentences)

        if truncate_sentences_p > random.random():
            in_sentences = in_sentences[:truncate_after_idx+1]

        out_sentences += in_sentences

        out_caption = ". ".join(out_sentences)

        if replace_dots_with_commas_p > random.random():
            out_caption = out_caption.replace(". ", ", ")

        if ending_dot_p > random.random():
            out_caption += "."

        out_caption = prefix + '. ' + out_caption

        #print(f"transformed caption from '{caption}' to '{out_caption}'")
        return out_caption

def partial_shuffle(l, factor=5):
    n = len(l)
    if n == 0:
        return
    for _ in range(factor):
        a, b = random.randrange(n), random.randrange(n)
        l[b], l[a] = l[a], l[b]
