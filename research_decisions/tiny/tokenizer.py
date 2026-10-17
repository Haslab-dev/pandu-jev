"""Domain-Specific Micro Tokenizer (2,048 Vocab Budget).

Provides fast subword/word-level tokenization for developer and agent prompts:
- Compact vocabulary of 2,048 tokens strictly bounded for <2MB model footprints
- Special tokens: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
- Character-level fallback for arbitrary identifier out-of-vocabulary terms
"""

import re
from typing import Dict, List, Optional
from datasets.rakitkode_domain import get_rakitkode_vocabulary

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
MAX_VOCAB_SIZE = 2048


class MicroTokenizer:
    """Ultra-lightweight domain tokenizer with a 2,048 token budget."""

    def __init__(self, vocab: Optional[List[str]] = None):
        self.special_tokens = SPECIAL_TOKENS
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.mask_token_id = 4

        # Build vocabulary
        vocab_list = list(SPECIAL_TOKENS)

        # Add single ASCII printable characters for fallback
        for ch in range(32, 127):
            char_str = chr(ch)
            if char_str not in vocab_list:
                vocab_list.append(char_str)

        # Add domain vocabulary words
        raw_vocab = vocab or get_rakitkode_vocabulary()
        for w in raw_vocab:
            w_norm = w.lower().strip()
            if w_norm and w_norm not in vocab_list and len(vocab_list) < MAX_VOCAB_SIZE:
                vocab_list.append(w_norm)

        # Fill remaining slots with common subwords and code tokens
        common_subwords = [
            "ing", "ed", "tion", "able", "ment", "ness", "ity", "ous", "al", "ic",
            "pre", "post", "sub", "un", "re", "de", "dis", "auto", "multi", "async",
            "get", "set", "is", "has", "do", "can", "with", "from", "into", "over",
            "true", "false", "int", "str", "bool", "float", "byte", "char", "void",
            "json", "yaml", "xml", "html", "css", "sql", "http", "https", "grpc",
            "api", "cli", "sdk", "tui", "gui", "ssh", "ssl", "tls", "jwt", "md5",
        ]
        for sw in common_subwords:
            if sw not in vocab_list and len(vocab_list) < MAX_VOCAB_SIZE:
                vocab_list.append(sw)

        self.id_to_token = vocab_list
        self.token_to_id = {tok: idx for idx, tok in enumerate(vocab_list)}
        self.vocab_size = len(vocab_list)

    def tokenize(self, text: str) -> List[str]:
        """Split text into words and sub-tokens."""
        # Split by punctuation and whitespace while keeping words
        raw_tokens = re.findall(r"__[\w]+__|[\w]+|[^\w\s]", text.lower())
        result = []
        for tok in raw_tokens:
            if tok in self.token_to_id:
                result.append(tok)
            else:
                # Character fallback for unknown identifiers
                for ch in tok:
                    result.append(ch if ch in self.token_to_id else "[UNK]")
        return result

    def encode(self, text: str, max_length: int = 256, add_special_tokens: bool = True) -> List[int]:
        """Encode text into a list of token IDs."""
        tokens = self.tokenize(text)
        if add_special_tokens:
            token_ids = [self.cls_token_id] + [self.token_to_id.get(t, self.unk_token_id) for t in tokens] + [self.sep_token_id]
        else:
            token_ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]

        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length-1] + [self.sep_token_id]
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to a string."""
        tokens = [self.id_to_token[i] if 0 <= i < self.vocab_size else "[UNK]" for i in token_ids]
        tokens = [t for t in tokens if t not in ["[PAD]", "[CLS]", "[SEP]"]]
        return " ".join(tokens)
