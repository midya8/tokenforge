# TokenForge

Byte-level BPE tokenizers trained from scratch, with no dependencies.

TokenForge trains byte pair encoding merges from raw text, encodes and decodes with exact round trips, and reports compression statistics as JSON or Markdown. It is dependency-free and runs fully offline, so a tokenizer can be diffed, reviewed, rebuilt byte for byte, and gated on in CI.

## Why

The tokenizer decides what a language model sees, but it usually ships as a black box artifact. TokenForge makes the whole pipeline inspectable:

- base vocabulary of the 256 byte values, so any text - ASCII, Unicode, emoji - encodes and decodes exactly;
- deterministic training: exact pair counts, ties broken on the pair itself, the same corpus always yields the same merges;
- pre-tokenization boundaries (words, numbers, symbol runs, whitespace) so merges never cross them, in the style of GPT-2;
- merges stored as plain JSON, reviewable and rebuildable;
- compression statistics (bytes per token, top tokens) with a `--min-compression` gate that fails CI when a tokenizer regresses.

## Commands

Train a tokenizer:

```bash
python3 -m tokenforge train --input corpus.txt --vocab-size 300 --output vocab.json
```

Encode and decode:

```bash
python3 -m tokenforge encode --vocab vocab.json --text "the tokenizer decides"
python3 -m tokenforge decode --vocab vocab.json --ids "1288 4210 6219"
```

Measure compression and gate on it:

```bash
python3 -m tokenforge stats --vocab vocab.json --input corpus.txt \
  --json stats.json --markdown stats.md --min-compression 1.2
```

## How it works

Training pre-tokenizes the corpus, counts adjacent token pairs across all pre-token sequences, and merges the most frequent pair into a new token id (starting at 256), repeating until the vocabulary reaches the requested size or no pair meets `--min-pair-count`. Encoding applies the same merges to each pre-token in rank order - always the lowest-rank available pair first - which is what keeps the token stream identical to the one training produced.

## Verification

- 40 unit tests cover pre-tokenization, pair counting and merging, deterministic training, vocab caps and minimum counts, encode/decode round trips across ASCII, Unicode and emoji, save/load, statistics math, Markdown escaping, and the CLI end to end including both sides of the compression gate.
- `bash examples/demo.sh` runs fully offline against the bundled corpus: it trains a 300-token vocabulary, verifies an encode/decode round trip, asserts the compression statistics, and checks both sides of the quality gate.
- CI runs the same suite across Python 3.10-3.13.

Byte-level BPE is the tokenizer family used by GPT-style models; this is a from-scratch teaching-and-tooling implementation, not a drop-in replacement for production vocabularies. The pre-tokenization pattern is ASCII-oriented and simplified relative to GPT-2's Unicode-aware split.

## License

MIT
