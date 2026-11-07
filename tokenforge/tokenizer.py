"""Byte-level BPE tokenizer trained from scratch.

The base vocabulary is the 256 byte values, so any byte sequence can be
encoded and decoded. Training counts adjacent token pairs across the
pre-tokenized corpus and merges the most frequent pair into a new token,
repeating until the vocabulary reaches the requested size or no pair meets
the minimum frequency. Ties are broken deterministically on the pair
itself, so training the same corpus twice yields the same merges.
"""

import json

FORMAT_VERSION = 1


def _pair_counts(seqs):
    counts = {}
    for seq in seqs:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge_seq(seq, pair, new_id):
    out = []
    i = 0
    n = len(seq)
    while i < n:
        if i + 1 < n and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE tokenizer with deterministic training."""

    def __init__(self, merges=None):
        self.merges = list(merges) if merges else []
        self._id_to_bytes = {i: bytes([i]) for i in range(256)}
        for idx, (a, b) in enumerate(self.merges):
            self._id_to_bytes[256 + idx] = self._id_to_bytes[a] + self._id_to_bytes[b]
        self._bytes_to_id = {v: k for k, v in self._id_to_bytes.items()}
        self._merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text, vocab_size=512, min_pair_count=2, pretokenizer=None):
        """Train merges from text. Returns a new BPETokenizer."""
        from .pretokenize import pretokenize

        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        if min_pair_count < 1:
            raise ValueError("min_pair_count must be at least 1")
        split = pretokenizer or pretokenize
        seqs = [list(p.encode("utf-8")) for p in split(text)]
        seqs = [s for s in seqs if s]
        merges = []
        while 256 + len(merges) < vocab_size:
            counts = _pair_counts(seqs)
            if not counts:
                break
            best_count = max(counts.values())
            if best_count < min_pair_count:
                break
            best = min(p for p, c in counts.items() if c == best_count)
            new_id = 256 + len(merges)
            seqs = [_merge_seq(s, best, new_id) for s in seqs]
            merges.append(best)
        return cls(merges)

    def encode(self, text):
        """Encode text to a list of token ids."""
        from .pretokenize import pretokenize

        ids = []
        for pre in pretokenize(text):
            seq = list(pre.encode("utf-8"))
            if not seq:
                continue
            while len(seq) > 1:
                best_rank = None
                best_pos = -1
                for i in range(len(seq) - 1):
                    rank = self._merge_rank.get((seq[i], seq[i + 1]))
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank = rank
                        best_pos = i
                if best_rank is None:
                    break
                pair = (seq[best_pos], seq[best_pos + 1])
                seq = _merge_seq(seq, pair, 256 + best_rank)
            ids.extend(seq)
        return ids

    def decode(self, ids):
        """Decode token ids back to text."""
        chunks = []
        for i in ids:
            piece = self._id_to_bytes.get(i)
            if piece is None:
                raise ValueError("unknown token id: %r" % (i,))
            chunks.append(piece)
        return b"".join(chunks).decode("utf-8", errors="replace")

    def token_bytes(self, token_id):
        return self._id_to_bytes[token_id]

    def save(self, path):
        payload = {"version": FORMAT_VERSION, "merges": [list(m) for m in self.merges]}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("version") != FORMAT_VERSION:
            raise ValueError("unsupported tokenizer format version: %r" % payload.get("version"))
        merges = [tuple(m) for m in payload["merges"]]
        return cls(merges)
