# tokenforge

A byte-level BPE tokenizer you can train on your own text in a few seconds, written in plain Python. I built it to understand how GPT-style tokenizers actually pick their merges, and it's small enough to read in one sitting.

```bash
python -m tokenforge train --input examples/data/corpus.txt --vocab-size 300 --output vocab.json

IDS=$(python -m tokenforge encode --vocab vocab.json --text "the tokenizer decides")
python -m tokenforge decode --vocab vocab.json --ids "$IDS"

python -m tokenforge stats --vocab vocab.json --input examples/data/corpus.txt --min-compression 1.2
```

On the bundled 1.2 KB corpus that learns 44 merges on top of the 256 byte tokens and gets about 1.57 bytes per token. "the tokenizer decides" comes out as 13 tokens and decodes back to the same string.

How it works:

- Text is split into pre-tokens (words, numbers, symbol runs, whitespace) and merges never cross those boundaries. The split regex is a simplified ASCII one, not GPT-2's.
- Every adjacent byte pair is counted, the most common one becomes a new token, and that repeats until the vocab is full or no pair shows up often enough (`--min-pair-count`). Ties go to the smaller pair so training is deterministic.
- Since every byte is in the base vocab, any UTF-8 text round-trips, emoji included.
- The vocab and merge list are saved as plain JSON.

A small thing I found interesting: in `aaaaa` the pair `aa` occurs four times but only two of them can be merged in the first pass, so pair count isn't the same as how much you compress.

There's a second trainer, `--trainer indexed`, that tracks pair positions with a heap instead of rescanning everything after each merge. It produces the same 44 merges and identical vocab bytes on the bundled corpus. INDEXED.md explains it, and https://midya8.github.io/tokenforge/ steps through a training run in the browser.

## Tests

`python -m unittest discover -s tests` (49 tests) and `bash examples/demo.sh`. MIT.
