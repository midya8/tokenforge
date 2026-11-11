# Same merges, different work

TokenForge 0.2 adds an optional occurrence-indexed trainer to the existing tokenizer.
The original full-recount `BPETokenizer.train` remains unchanged and is still the
CLI default. The original byte vocabulary, regex split, rank encoding, saved
format, corpus, demo, statistics and 40 tests are retained. This is not another
tokenizer project, a replacement production vocabulary or a new BPE algorithm.

## The semantic contract comes before the data structure

Every pre-token is an independent UTF-8 byte row. Initial token ids are 0..255.
At each training iteration, count **all adjacent occurrences, including overlaps**
across rows but never across a row boundary. Choose highest frequency, breaking
ties by the smallest numeric `(a,b)` token pair. Stop at the requested vocabulary
cap, no pairs, or when the maximum count is below the supplied minimum. New ids
are 256 plus merge rank. Replace every leftmost nonoverlapping occurrence of the
selected pair within each row. A five-byte `aaaaa` row has four `aa` occurrences,
but the first merge makes only two replacements at original left positions 0,2,
giving `[256,256,97]`.

Frequency and compression gain differ under overlap. This follows the original
TokenForge behavior. It is not the nonoverlap-adjusted objective in parts of
Zouhar and colleagues' formal analysis, so their optimization/complexity theorem
is not transferred to this implementation. Exact ties are part of this project's
contract, not a claim that all BPE libraries choose the same merges.

## Occurrences, linked positions and a lazy heap

Each original byte position has a token id, previous/next links and a live flag.
Positions remain in original order; a merge keeps its left position and unlinks
its right one. Each live adjacency appears in exactly one pair's set of left
positions. Initializing this index visits every input adjacency once.

For each chosen pair, snapshot and sort its live positions. At each position,
check that the pair is still present. A prior replacement may have removed an
overlapping occurrence. Remove the preceding edge, selected edge and following
edge where present; relink and insert the new preceding/following edges. Counts
are set sizes. After the whole round, push the affected nonempty pairs with
`(-count,a,b)` priorities. Deleted pairs leave stale heap entries. Pop until the
entry's count agrees with the current set size. A stale entry whose count happens
to agree again is a valid current priority; pair identity and tie key also match.
There is no monotonic-count assumption and no generation id is required.

The invariant is that pair sets are exactly the current live adjacencies. Local
edge edits preserve it. A valid heap maximum selects the same frequency/tie pair
as recounting. Sorted position processing with validity checks matches the old
left-to-right nonoverlap replacement. These arguments describe intended algorithm
behavior. The bounded independent tests are evidence of the code, not a formal
proof of Python's implementation or unbounded universal equivalence.

```python
from tokenforge import train_incremental

tokenizer, report = train_incremental(text, vocab_size=300, min_pair_count=2)
# The same BPETokenizer object format and encoder are used.
ids = tokenizer.encode(text)
assert tokenizer.decode(ids) == text
```

`trace=False` is the default. `trace=True` copies all current pair counts and
before/after rows each selected iteration. It also records original left-position
ids, overlapping frequency and actual replacement count. Trace copying is extra
work, not part of an alleged speed benchmark. The input rows and returned results
are detached. A run trains a new model from scratch. It is not streaming training,
online corpus updates, a saved index or continued vocabulary learning.

```sh
python -m tokenforge train --input examples/data/corpus.txt \
  --vocab-size 300 --trainer indexed --output vocab.json \
  --training-report training.json
```

The optional training report requires indexed mode. Without `--trainer indexed`,
training still runs the original full scan. Existing command output and saved
vocabulary JSON are unchanged for the same valid training request.

## Work witness, not elapsed time or RAM

For the original 1,227-byte bundled corpus at requested vocabulary 300 and minimum
pair count 2, both trainers produce the same 44 merges and identical saved JSON.
The original scan's recount function visits 30,664 adjacent positions, verified
by instrumenting that original function in a test. The indexed trainer records:

|Selected work|Count|
|---|---:|
|Initial adjacency insertions|985|
|Edge removals|1,014|
|New edge insertions|568|
|Snapshot positions visited|446|
|Actual pair replacements|446|
|Heap pushes|652|
|Heap pops|117|
|Of those, stale pops|73|

These columns count different operations and must not be subtracted or combined
into a speedup. The original scan also constructs counts, selects maximum and
rewrites rows. The index allocates link arrays, occurrence sets, changed-pair
sets, snapshots, sorting and heap entries. This implementation does not compact
the lazy heap; stale entries can accumulate until the run ends. Dictionary/set
and integer costs, cache behavior and actual allocation sizes are not measured.
Tracing that corpus additionally copies 13,857 pair records and 82,178 token
entries. A low update count is not a timing, RAM, total-operation or proven
complexity result. Small datasets can favor the simpler full scan.

## Bounded audit and independent browser

The audit input has exactly `rows`, `vocab_size`, `min_pair_count`. Rows contain
nonboolean integer bytes 0..255, including zero. Empty rows are retained and never
joined. Limits: at most 64 rows, 1,024 bytes, vocabulary 256..288, minimum 1..1,024,
and 128 KiB UTF-8 JSON. Duplicate keys, floats in integer fields, unknown keys and
invalid bounds fail clearly. The reusable native trainer has no 1,024-byte limit;
its arguments still require exact integer byte ids and valid counts/caps.

```sh
python -m tokenforge.incremental_cli --input input.json > native.json
tokenforge-indexed --input input.json > native.json
```

Native output separates `semantic` from `indexed_work`. The semantic report
contains every selected pair, count, replacement position, complete pair table,
complete before/after rows, final rows and stop reason. The browser independently
recounts every adjacency and constructs fresh output rows, carrying original
left-position ids. It does not use Python linked lists, occurrence sets, heap
code or shipped native reports. Only witness inputs are shipped to the page.

The browser exports the same input and an independently computed semantic report.
To replay an export, run the downloaded input through either native command and
compare `input` and `semantic` exactly. The browser report schema is deliberately
`tokenforge-indexed-semantic/1`, versus native `tokenforge-indexed/1`. Native work
counters are **not** browser-verified, bundled into the semantic export or claimed
as independently replayed there. The browser's reference adjacency visits match
the native scan-reference counter, but they do not validate heap operations.

The original-first-twelve control uses the original Python regex's first 12
pre-token rows. The Unicode control uses explicit UTF-8 rows. The page does not
claim cross-language regex/Unicode splitting parity. Edits hide stale results and
disable downloads. All data stays local; there is no upload or model API.

## Independent checks and reproduction

- 49 Python tests retain all 40 original tests unchanged. Independent full recount
  and fresh-row replacement covers all binary rows of length 0..8 under 5 cap/count
  settings: 2,555 comparisons. Another 250 seeded multirow tables and 39 overlapping
  run/boundary controls compare every merge, pair table and row, not only finals.
- 28 original-scan/indexed text settings cover the full bundled corpus, empty input,
  ASCII, whitespace, repeated text, UTF-8/emoji and mixed scripts. Merge lists,
  token bytes, encoded ids, decoded text and saved vocabulary bytes match.
- 27 complete new audit reports include overlaps, equal-count ties, separated
  singletons, empty rows, byte extremes, stop conditions, original rows, UTF-8
  rows and 16 seeded inputs. Their 160 selected merge steps retain full witnesses.
  Conservation checks verify initial+inserted-removed edge counts and token loss.
- 4 Node/actual-Chrome tests compare all 27 full semantic reports and independently
  repeat 2,555 small-row/cap/count comparisons. Actual Chrome inspects all 160 merge
  choices and 3,033 pair-table rows, 24 responsive views, invalidation/strict errors
  and 2 trusted input/semantic downloads. No approximate tolerance is used.
-An eight-file installed package is checked outside the checkout with 7 imported
  module origins. All 27 audits, the original corpus's 44 merge parity, saved JSON,
  encode/decode/stats, indexed train+trace, module and console bounded CLI pass.
  Original offline demo and both compression-gate sides remain unchanged.

These are development fixtures, not a representative language corpus evaluation,
compression optimum, training-quality or runtime benchmark. All original source
history and default scan are preserved. BPE greedy frequency is not a global
minimum-token guarantee.

## Sources and differences

- [Gage1994, A New Algorithm for Data Compression](https://www.derczynski.com/papers/archive/BPE_Gage.pdf):
  established pair substitution/compression origin. This project does not copy its
  file-format/code table design or claim a new compressor.
- [Sennrich, Haddow and Birch2016](https://aclanthology.org/P16-1162.pdf):
  established BPE subword application, distinct from this simplified byte regex.
- [Reference subword-nmt implementation](https://github.com/rsennrich/subword-nmt/blob/92d6139d07d30e12735a0af9e7f7f925ebe62c54/subword_nmt/learn_bpe.py):
  existing incremental pair-statistics updates; different vocabulary, tie and
  weighting conventions are not asserted equivalent. No implementation copied.
- [Zouhar and colleagues, A Formal Perspective on Byte-Pair Encoding](https://arxiv.org/html/2306.16837):
  linked-list/occurrence/priority-queue prior work and explicit overlap distinction.
  Its optimized top-M queue and nonoverlap-adjusted objective/theorems are not
  this code's algorithm or guarantees.
- [Gowda2026, Faster BPE training](https://gowda.ai/posts/2026/03/faster-bpe-learn/):
  linked positions, max-heap and lazy deletion implementation discussion. Its
  production benchmarks/speed claims are not measurements of this backend.
- [Tan2025, Building a Fast BPE Tokenizer from Scratch](https://jytan.net/blog/2025/bpe/):
  implementation tradeoffs, heap overhead and stale-entry memory risk. This
  backend deliberately makes no crossover or performance claim.
- [Python heapq documentation](https://docs.python.org/3/library/heapq.html):
  min-heap/tuple ordering and changed-priority stale-entry patterns. Python 3.10
  compatibility uses negative counts rather than newer max-heap functions.
