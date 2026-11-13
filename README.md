# TokenForge

Train a byte-level BPE tokenizer, inspect its merges, and check that text survives
an encode/decode round trip. TokenForge runs offline in dependency-free Python.
Saved vocabularies are plain JSON; compression reports are JSON or Markdown.

## Two trainers, the same merge rules

The original scan is still the default. `--trainer indexed` trains the same
byte-level BPE model using live occurrence sets, linked positions and a lazy heap.
On the bundled corpus, both trainers produce the same 44 merges and saved
vocabulary bytes. Overlapping pair frequencies, numeric ties and leftmost
replacement order are preserved.

[Interactive training witness](https://midya8.github.io/tokenforge/) |
[Semantic/work contract and sources](INDEXED.md)

The independent browser recounts complete merge histories on explicit byte rows.
It shows why `aaaaa` has four adjacent `aa` matches but only two first-round
replacements. Edit inputs, inspect every pair table and row, and export replayable
input/semantic JSON. Native work counters are structural, not latency/RAM results;
the browser does not execute or validate the native heap.

```sh
python -m tokenforge train --input examples/data/corpus.txt \
  --vocab-size 300 --trainer indexed --output vocab.json \
  --training-report training.json
```

49 Python tests retain all 40 originals; 4 Node/actual-Chrome tests check full
histories, 3,033 displayed pair rows, 24 responsive views and trusted exports.
Installed wheel checks run outside the checkout. This is an optional backend in
TokenForge, not a new tokenizer project or a production performance claim.

## Model format

The model exposes its vocabulary and ordered merges:

- base vocabulary of the 256 byte values, so encodable UTF-8 text, including Unicode and emoji, round-trips;
- deterministic training: exact pair counts, ties broken on the pair itself, the same corpus always yields the same merges;
- pre-tokenization boundaries (words, numbers, symbol runs, whitespace) that
  merges never cross; the regex is ASCII-oriented, not GPT-2 splitting parity;
- merges stored as plain JSON, reviewable and rebuildable;
- compression statistics (bytes per token, top tokens) with a `--min-compression` gate that fails when the measured corpus compression
  falls below the supplied threshold. It is not a language-model quality test.

## Commands

Train a tokenizer:

```bash
python3 -m tokenforge train --input examples/data/corpus.txt --vocab-size 300 --output vocab.json
```

Encode and decode:

```bash
IDS=$(python3 -m tokenforge encode --vocab vocab.json --text "the tokenizer decides")
python3 -m tokenforge decode --vocab vocab.json --ids "$IDS"
```

Measure compression and gate on it:

```bash
python3 -m tokenforge stats --vocab vocab.json --input examples/data/corpus.txt \
  --json stats.json --markdown stats.md --min-compression 1.2
```

## Counting and replacing pairs

Each pre-token becomes a separate UTF-8 byte row. Training counts every adjacent
pair, including overlaps, without crossing rows. It picks the highest count,
breaks ties by the smallest numeric token pair, and replaces leftmost
nonoverlapping occurrences with a new token id, starting at 256. It stops at the
requested vocabulary cap, when no pairs remain, or below `--min-pair-count`.

For `aaaaa`, the pair `aa` has four occurrences but only two replacements in the
first round. Pair frequency is not the same as compression gain. The
[training witness](https://midya8.github.io/tokenforge/) lets you inspect that
choice and every row after it.

Encoding starts from the same byte rows and repeatedly applies the lowest-rank
available learned pair. Decoding joins the stored token bytes, then decodes UTF-8.
Text encoded by the tokenizer round-trips; an arbitrary list of valid token ids
can split UTF-8 characters, in which case decoding uses replacement characters.
Unknown token ids fail.

## Verification

- The original 40 unit tests cover pre-tokenization, pair counting and merging, deterministic training, vocab caps and minimum counts, encode/decode round trips across ASCII, Unicode and emoji, save/load, statistics math, Markdown escaping, and the CLI end to end including both sides of the compression gate.
- `bash examples/demo.sh` runs fully offline against the bundled corpus: it trains a 300-token vocabulary, verifies an encode/decode round trip, asserts the compression statistics, and checks both sides of the compression gate.
- CI runs 49 Python tests across Python 3.10-3.13, plus witness regeneration, installed wheel reproduction and 4 independent Node/Chrome tests.

## Scope

This is an offline implementation of established BPE, not a production vocabulary,
a new tokenizer algorithm or a language-model training result. The pre-tokenizer
uses a simplified ASCII-oriented regex rather than GPT-2's Unicode-aware split.
The indexed backend is optional; the original scan, format and defaults remain.
Work counters describe different operations, not a measured speedup or memory
saving. See [the method and source notes](INDEXED.md) for the lazy-heap invariant,
bounded checks and comparisons with prior implementations.

## License

MIT
