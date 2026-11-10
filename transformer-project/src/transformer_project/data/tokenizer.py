"""Thin, typed wrapper around HuggingFace `tokenizers` with special-token handling."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

logger = logging.getLogger(__name__)

# Order matters: the first six keep the ids they always had (the post-processor
# below refers to BOS=1, EOS=2, SEP=4), so older tokenizers/checkpoints stay
# compatible. New structural tokens are appended after them so that source
# formats (commits, Q&A, issues, notebooks) are single tokens, not fragments.
SPECIAL_TOKENS = [
    "[UNK]", "[BOS]", "[EOS]", "[PAD]", "[SEP]", "[MASK]",
    "<file_sep>",
    "<commit_before>", "<commit_message>", "<commit_after>",
    "<question>", "<answer>",
    "<issue_start>", "<issue_comment>",
    "<jupyter_text>", "<jupyter_code>", "<jupyter_output>",
]


class TokenizerWrapper:
    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer
        self.bos_id = self.tok.token_to_id("[BOS]")
        self.eos_id = self.tok.token_to_id("[EOS]")
        self.pad_id = self.tok.token_to_id("[PAD]")
        self.unk_id = self.tok.token_to_id("[UNK]")

    @classmethod
    def train(cls, files: List[str], vocab_size: int, save_path: str) -> "TokenizerWrapper":
        if not files:
            raise ValueError("No training files provided to TokenizerWrapper.train()")
        missing = [f for f in files if not Path(f).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} training file(s) not found, e.g. {missing[:3]}")

        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        # ByteLevel: every byte is representable (no [UNK] on unusual unicode
        # or symbols in code), and it preserves leading whitespace/indentation,
        # which matters for languages like Python where indentation is
        # semantically meaningful.
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [SEP] $B:1 [EOS]:1",
            special_tokens=[("[BOS]", 1), ("[EOS]", 2), ("[SEP]", 4)],
        )
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, min_frequency=2, show_progress=True
        )
        tok.train(files, trainer)

        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(save_path_obj))
        logger.info("Tokenizer trained and saved to %s (vocab_size=%d)", save_path, vocab_size)
        return cls(tok)

    @classmethod
    def load(cls, path: str) -> "TokenizerWrapper":
        if not Path(path).exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        tok = Tokenizer.from_file(path)
        logger.info("Tokenizer loaded from %s (vocab_size=%d)", path, tok.get_vocab_size())
        return cls(tok)

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        return self.tok.decode(ids, skip_special_tokens=skip_special)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()
