"""HF-like adapter around a miditok tokenizer for the ELF training pipeline."""

from pathlib import Path

from miditok import REMI, MusicTokenizer


def _tokenize_ids(tok, midi_path):
    """Return flat token ids from miditok (list of TokSequence or TokSequence)."""
    out = tok(midi_path)
    if isinstance(out, list):
        return [i for seq in out for i in seq.ids]
    return out.ids


class MusicTokenizerAdapter:
    """Expose miditok special tokens and vocab size in an HF-tokenizer-like surface."""

    def __init__(self, miditok_tok: MusicTokenizer):
        self.tok = miditok_tok

    def __len__(self) -> int:
        return len(self.tok)

    @property
    def vocab_size(self) -> int:
        return len(self.tok)

    @property
    def pad_token_id(self) -> int:
        if hasattr(self.tok, "pad_token_id") and self.tok.pad_token_id is not None:
            return self.tok.pad_token_id
        return self.tok["PAD_None"]

    @property
    def eos_token_id(self):
        if hasattr(self.tok, "eos_token_id") and self.tok.eos_token_id is not None:
            return self.tok.eos_token_id
        try:
            return self.tok["EOS_None"]
        except (KeyError, ValueError):
            return None

    def decode(self, ids, **kwargs):
        return self.tok(ids, **kwargs)


def get_music_tokenizer(path: str) -> MusicTokenizerAdapter:
    """Load a saved miditok tokenizer and wrap it."""
    tok = REMI(params=str(Path(path)))
    return MusicTokenizerAdapter(tok)
