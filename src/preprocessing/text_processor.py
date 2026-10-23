"""
Text Modality Preprocessor
Handles text normalization, cleaning, tokenization, and padding/truncation
as described in Section V-E of the research paper.
"""

import re
import numpy as np

class TextPreprocessor:
    def __init__(self, max_length=64, vocab_size=10000):
        self.max_length = max_length
        self.vocab_size = vocab_size
        # Special tokens
        self.PAD_TOKEN = "[PAD]"
        self.UNK_TOKEN = "[UNK]"
        self.CLS_TOKEN = "[CLS]"
        self.SEP_TOKEN = "[SEP]"

        self.token_to_id = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.CLS_TOKEN: 2,
            self.SEP_TOKEN: 3
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def clean_text(self, text):
        """
        Normalizes text: lowercase, expands contractions, removes abnormal characters.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        text = text.lower().strip()
        # Common contractions
        text = re.sub(r"can\'t", "cannot", text)
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)

        # Retain punctuation meaningful for emotion (exclamation, question marks)
        text = re.sub(r"[^a-zA-Z0-9\s!\?.,]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text):
        """
        Tokenizes text into words and punctuation marks.
        """
        cleaned = self.clean_text(text)
        tokens = re.findall(r"\w+|[^\w\s]", cleaned)
        return tokens

    def build_vocab_from_texts(self, texts):
        """
        Builds vocabulary dictionary from a corpus of texts.
        """
        freq = {}
        for text in texts:
            tokens = self.tokenize(text)
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1

        sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        for token, _ in sorted_tokens:
            if len(self.token_to_id) >= self.vocab_size:
                break
            if token not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[token] = new_id
                self.id_to_token[new_id] = token

    def encode(self, text):
        """
        Converts text string into input_ids and attention_mask tensors.
        Returns:
            input_ids: numpy array [max_length]
            attention_mask: numpy array [max_length]
        """
        tokens = self.tokenize(text)
        # Truncate if necessary (accounting for [CLS] and [SEP])
        available_slots = self.max_length - 2
        tokens = tokens[:available_slots]

        seq = [self.token_to_id[self.CLS_TOKEN]]
        for t in tokens:
            seq.append(self.token_to_id.get(t, self.token_to_id[self.UNK_TOKEN]))
        seq.append(self.token_to_id[self.SEP_TOKEN])

        length = len(seq)
        pad_len = self.max_length - length

        input_ids = seq + [self.token_to_id[self.PAD_TOKEN]] * pad_len
        attention_mask = [1] * length + [0] * pad_len

        return (
            np.array(input_ids, dtype=np.int64),
            np.array(attention_mask, dtype=np.float32)
        )
