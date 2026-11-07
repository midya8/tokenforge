#!/usr/bin/env bash
# Fully offline TokenForge demo: train a tokenizer on the bundled corpus,
# verify a round trip, compute compression statistics, and check the gate.
set -u
cd "$(dirname "$0")/.."

OUT="$(mktemp -d /tmp/tokenforge-demo-XXXX)"
trap 'echo "demo artifacts: $OUT"' EXIT

echo "== train =="
python3 -m tokenforge train \
  --input examples/data/corpus.txt \
  --vocab-size 300 \
  --output "$OUT/vocab.json"

echo "== round trip =="
TEXT="the tokenizer decides what the tokens are"
IDS=$(python3 -m tokenforge encode --vocab "$OUT/vocab.json" --text "$TEXT")
BACK=$(python3 -m tokenforge decode --vocab "$OUT/vocab.json" --ids "$IDS")
echo "encoded: $IDS"
echo "decoded: $BACK"
[ "$BACK" = "$TEXT" ] || { echo "FAIL round trip mismatch"; exit 1; }

echo "== stats =="
python3 -m tokenforge stats \
  --vocab "$OUT/vocab.json" \
  --input examples/data/corpus.txt \
  --json "$OUT/stats.json" \
  --markdown "$OUT/stats.md"

python3 - "$OUT/stats.json" <<'PY'
import json, sys
stats = json.load(open(sys.argv[1]))
assert stats["tokens"] > 0, "no tokens produced"
assert stats["bytes_per_token"] > 1.0, "no compression achieved"
assert stats["vocab_size"] <= 300, "vocab cap breached"
print("assertions ok: %d tokens, %.3f bytes/token, vocab %d"
      % (stats["tokens"], stats["bytes_per_token"], stats["vocab_size"]))
PY

echo "== gate (expect pass) =="
python3 -m tokenforge stats --vocab "$OUT/vocab.json" --input examples/data/corpus.txt \
  --min-compression 1.0 >/dev/null || { echo "FAIL gate that should pass"; exit 1; }
echo "gate passed as expected"

echo "== gate (expect fail) =="
if python3 -m tokenforge stats --vocab "$OUT/vocab.json" --input examples/data/corpus.txt \
  --min-compression 99.0 >/dev/null 2>&1; then
  echo "FAIL gate that should fail"; exit 1
fi
echo "gate failed as expected"

echo "demo ok"
