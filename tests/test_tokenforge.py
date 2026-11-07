import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tokenforge import BPETokenizer, corpus_stats, pretokenize
from tokenforge.stats import render_markdown, render_terminal
from tokenforge.tokenizer import _merge_seq, _pair_counts


class TestPretokenize(unittest.TestCase):
    def test_words_and_spaces(self):
        self.assertEqual(pretokenize("hello world"), ["hello", " world"])

    def test_numbers_split_from_words(self):
        self.assertEqual(pretokenize("abc123"), ["abc", "123"])

    def test_punctuation_runs(self):
        self.assertEqual(pretokenize("wait... really?!"), ["wait", "...", " really", "?!"])

    def test_contractions(self):
        self.assertEqual(pretokenize("it's they'll"), ["it", "'s", " they", "'ll"])

    def test_trailing_whitespace(self):
        self.assertEqual(pretokenize("a \n"), ["a", " \n"])

    def test_empty(self):
        self.assertEqual(pretokenize(""), [])

    def test_rejects_non_str(self):
        with self.assertRaises(TypeError):
            pretokenize(b"bytes")


class TestPairHelpers(unittest.TestCase):
    def test_pair_counts(self):
        self.assertEqual(_pair_counts([[1, 2, 1, 2]]), {(1, 2): 2, (2, 1): 1})

    def test_pair_counts_empty(self):
        self.assertEqual(_pair_counts([[]]), {})

    def test_merge_seq(self):
        self.assertEqual(_merge_seq([1, 2, 1, 2, 3], (1, 2), 9), [9, 9, 3])

    def test_merge_seq_no_overlap(self):
        self.assertEqual(_merge_seq([1, 1, 1], (1, 1), 9), [9, 1])


class TestTraining(unittest.TestCase):
    def test_vocab_size_floor(self):
        with self.assertRaises(ValueError):
            BPETokenizer.train("text", vocab_size=100)

    def test_min_pair_count_floor(self):
        with self.assertRaises(ValueError):
            BPETokenizer.train("text", min_pair_count=0)

    def test_base_vocab_without_merges(self):
        tok = BPETokenizer.train("no repeated pairs here", vocab_size=256)
        self.assertEqual(tok.vocab_size, 256)
        self.assertEqual(tok.merges, [])

    def test_merges_most_frequent_first(self):
        tok = BPETokenizer.train("abab abab", vocab_size=257, min_pair_count=1)
        self.assertEqual(len(tok.merges), 1)
        pair = tok.merges[0]
        self.assertEqual(tok.token_bytes(256), tok.token_bytes(pair[0]) + tok.token_bytes(pair[1]))

    def test_training_is_deterministic(self):
        text = "the cat sat on the mat. the cat ate the rat."
        a = BPETokenizer.train(text, vocab_size=270)
        b = BPETokenizer.train(text, vocab_size=270)
        self.assertEqual(a.merges, b.merges)

    def test_training_stops_at_min_count(self):
        tok = BPETokenizer.train("abcdefg", vocab_size=300, min_pair_count=2)
        self.assertEqual(tok.merges, [])

    def test_vocab_size_cap_respected(self):
        tok = BPETokenizer.train("aa bb aa bb " * 40, vocab_size=260)
        self.assertLessEqual(tok.vocab_size, 260)

    def test_merges_never_cross_pretokens(self):
        tok = BPETokenizer.train("ab ab ab", vocab_size=300, min_pair_count=1)
        for a, b in tok.merges:
            piece = tok.token_bytes(256) if False else None
        # 'b a' across the space is never a merge because space leads the next pretoken
        self.assertNotIn(b"b a", [tok.token_bytes(256 + i) for i in range(len(tok.merges))])


class TestEncodeDecode(unittest.TestCase):
    def setUp(self):
        self.text = "the quick brown fox jumps over the lazy dog. " * 5
        self.tok = BPETokenizer.train(self.text, vocab_size=280)

    def test_round_trip_ascii(self):
        ids = self.tok.encode(self.text)
        self.assertEqual(self.tok.decode(ids), self.text)

    def test_round_trip_unicode(self):
        text = "café — naïve — 東京 — 🚀"
        ids = self.tok.encode(text)
        self.assertEqual(self.tok.decode(ids), text)

    def test_empty_text(self):
        self.assertEqual(self.tok.encode(""), [])
        self.assertEqual(self.tok.decode([]), "")

    def test_all_byte_values_encodable(self):
        text = "".join(chr(i) for i in range(1, 128))
        ids = self.tok.encode(text)
        self.assertEqual(self.tok.decode(ids), text)

    def test_compression_improves_with_merges(self):
        merged = self.tok.encode(self.text)
        plain = BPETokenizer().encode(self.text)
        self.assertLess(len(merged), len(plain))

    def test_encode_uses_lowest_rank_first(self):
        tok = BPETokenizer.train("aaaa", vocab_size=258, min_pair_count=1)
        # token 256 is "aa", token 257 is "aaaa"; both merges apply greedily by rank
        self.assertEqual(tok.encode("aaaa"), [257])
        self.assertEqual(tok.encode("aaa"), [256, 97])

    def test_decode_unknown_id(self):
        with self.assertRaises(ValueError):
            self.tok.decode([9999])

    def test_untrained_tokenizer_is_byte_level(self):
        tok = BPETokenizer()
        ids = tok.encode("abc")
        self.assertEqual(ids, [97, 98, 99])


class TestSaveLoad(unittest.TestCase):
    def test_round_trip(self):
        tok = BPETokenizer.train("the cat the cat the cat", vocab_size=270)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            tok.save(path)
            loaded = BPETokenizer.load(path)
        self.assertEqual(loaded.merges, tok.merges)
        self.assertEqual(loaded.encode("the cat"), tok.encode("the cat"))

    def test_load_rejects_bad_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            path.write_text(json.dumps({"version": 99, "merges": []}))
            with self.assertRaises(ValueError):
                BPETokenizer.load(path)


class TestStats(unittest.TestCase):
    def setUp(self):
        self.text = "the cat sat on the mat " * 8
        self.tok = BPETokenizer.train(self.text, vocab_size=280)

    def test_stats_math(self):
        stats = corpus_stats(self.tok, self.text)
        self.assertEqual(stats["bytes"], len(self.text.encode("utf-8")))
        self.assertEqual(stats["tokens"], len(self.tok.encode(self.text)))
        self.assertAlmostEqual(
            stats["bytes_per_token"], stats["bytes"] / stats["tokens"], places=6
        )
        self.assertEqual(stats["vocab_size"], self.tok.vocab_size)

    def test_stats_empty_text(self):
        stats = corpus_stats(self.tok, "")
        self.assertEqual(stats["tokens"], 0)
        self.assertEqual(stats["bytes_per_token"], 0.0)

    def test_top_tokens_capped(self):
        stats = corpus_stats(self.tok, self.text, top=3)
        self.assertEqual(len(stats["top_tokens"]), 3)

    def test_markdown_report(self):
        md = render_markdown(corpus_stats(self.tok, self.text))
        self.assertIn("# TokenForge corpus statistics", md)
        self.assertIn("| bytes |", md)
        self.assertIn("Most frequent tokens", md)

    def test_markdown_escapes_pipes(self):
        stats = corpus_stats(self.tok, "a|b a|b")
        md = render_markdown(stats)
        self.assertIn("\\|", md)

    def test_terminal_report(self):
        out = render_terminal(corpus_stats(self.tok, self.text))
        self.assertIn("compression", out)
        self.assertIn("vocab size", out)


class TestCli(unittest.TestCase):
    def run_cli(self, *args, cwd=None):
        return subprocess.run(
            [sys.executable, "-m", "tokenforge", *args],
            capture_output=True,
            text=True,
            cwd=cwd,
        )

    def test_train_encode_decode_stats_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = tmp / "corpus.txt"
            corpus.write_text("the quick brown fox " * 30)
            vocab = tmp / "vocab.json"

            r = self.run_cli(
                "train", "--input", str(corpus), "--vocab-size", "270",
                "--output", str(vocab),
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(vocab.exists())

            r = self.run_cli("encode", "--vocab", str(vocab), "--text", "the fox")
            self.assertEqual(r.returncode, 0, r.stderr)
            ids = r.stdout.strip()

            r = self.run_cli("decode", "--vocab", str(vocab), "--ids", ids)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), "the fox")

            report = tmp / "stats.json"
            md = tmp / "stats.md"
            r = self.run_cli(
                "stats", "--vocab", str(vocab), "--input", str(corpus),
                "--json", str(report), "--markdown", str(md),
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            data = json.loads(report.read_text())
            self.assertEqual(data["bytes"], len(corpus.read_text().encode("utf-8")))
            self.assertIn("TokenForge", md.read_text())

    def test_stats_gate_pass_and_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = tmp / "corpus.txt"
            corpus.write_text("the quick brown fox " * 30)
            vocab = tmp / "vocab.json"
            self.run_cli("train", "--input", str(corpus), "--output", str(vocab))

            r = self.run_cli(
                "stats", "--vocab", str(vocab), "--input", str(corpus),
                "--min-compression", "1.0",
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            r = self.run_cli(
                "stats", "--vocab", str(vocab), "--input", str(corpus),
                "--min-compression", "99.0",
            )
            self.assertEqual(r.returncode, 1)
            self.assertIn("FAIL", r.stderr)

    def test_encode_json_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = tmp / "corpus.txt"
            corpus.write_text("hello world hello world")
            vocab = tmp / "vocab.json"
            out = tmp / "ids.json"
            self.run_cli("train", "--input", str(corpus), "--output", str(vocab))
            r = self.run_cli(
                "encode", "--vocab", str(vocab), "--text", "hello", "--json", str(out)
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            data = json.loads(out.read_text())
            self.assertEqual(data["count"], len(data["tokens"]))

    def test_train_rejects_missing_input(self):
        r = self.run_cli("train", "--input", "/nonexistent.txt", "--output", "/tmp/x.json")
        self.assertNotEqual(r.returncode, 0)

    def test_decode_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus = tmp / "corpus.txt"
            corpus.write_text("abc abc abc")
            vocab = tmp / "vocab.json"
            idfile = tmp / "ids.txt"
            self.run_cli("train", "--input", str(corpus), "--output", str(vocab))
            r = self.run_cli("encode", "--vocab", str(vocab), "--text", "abc")
            idfile.write_text(r.stdout)
            r = self.run_cli("decode", "--vocab", str(vocab), "--input", str(idfile))
            self.assertEqual(r.stdout.strip(), "abc")


if __name__ == "__main__":
    unittest.main()
