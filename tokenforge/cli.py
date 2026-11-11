"""Command line interface for TokenForge."""

import argparse
import json
import sys

from .stats import corpus_stats, render_markdown, render_terminal, write_json_report
from .tokenizer import BPETokenizer


def _read_text(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _cmd_train(args):
    text = _read_text(args.input)
    if args.trainer == 'indexed':
        from .incremental import train_incremental
        tokenizer, report = train_incremental(text, args.vocab_size, args.min_pair_count, trace=bool(args.training_report))
        if args.training_report:
            with open(args.training_report, 'w', encoding='utf-8') as fh:
                json.dump(report, fh, indent=2); fh.write('\n')
    else:
        if args.training_report:
            raise ValueError('--training-report requires --trainer indexed')
        tokenizer = BPETokenizer.train(text, vocab_size=args.vocab_size, min_pair_count=args.min_pair_count)
    tokenizer.save(args.output)
    print(
        "trained %d merges (vocab size %d) -> %s"
        % (len(tokenizer.merges), tokenizer.vocab_size, args.output)
    )
    return 0


def _cmd_encode(args):
    tokenizer = BPETokenizer.load(args.vocab)
    text = args.text if args.text is not None else _read_text(args.input)
    ids = tokenizer.encode(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"tokens": ids, "count": len(ids)}, fh, indent=2)
            fh.write("\n")
    print(" ".join(str(i) for i in ids))
    return 0


def _cmd_decode(args):
    tokenizer = BPETokenizer.load(args.vocab)
    raw = args.ids if args.ids is not None else _read_text(args.input)
    ids = [int(part) for part in raw.replace(",", " ").split()]
    print(tokenizer.decode(ids))
    return 0


def _cmd_stats(args):
    tokenizer = BPETokenizer.load(args.vocab)
    text = _read_text(args.input)
    stats = corpus_stats(tokenizer, text)
    print(render_terminal(stats))
    if args.json:
        write_json_report(stats, args.json)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(render_markdown(stats))
    if args.min_compression is not None and stats["bytes_per_token"] < args.min_compression:
        print(
            "FAIL bytes/token %.4f below gate %.4f"
            % (stats["bytes_per_token"], args.min_compression),
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="tokenforge",
        description="Train and inspect byte-level BPE tokenizers from scratch.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train merges from a text corpus")
    p.add_argument("--input", required=True, help="training corpus (UTF-8 text)")
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--min-pair-count", type=int, default=2)
    p.add_argument("--output", required=True, help="output vocab JSON path")
    p.add_argument('--trainer', choices=['scan','indexed'], default='scan', help='scan remains the default; indexed updates local occurrences')
    p.add_argument('--training-report', help='indexed trace and structural work, not a timing report')
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("encode", help="encode text to token ids")
    p.add_argument("--vocab", required=True, help="vocab JSON from train")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--input")
    p.add_argument("--json", help="optional JSON output path")
    p.set_defaults(func=_cmd_encode)

    p = sub.add_parser("decode", help="decode token ids back to text")
    p.add_argument("--vocab", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ids", help="space or comma separated ids")
    src.add_argument("--input", help="file containing ids")
    p.set_defaults(func=_cmd_decode)

    p = sub.add_parser("stats", help="corpus compression statistics")
    p.add_argument("--vocab", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--json", help="write JSON report")
    p.add_argument("--markdown", help="write Markdown report")
    p.add_argument(
        "--min-compression",
        type=float,
        default=None,
        help="fail unless bytes/token reaches this value",
    )
    p.set_defaults(func=_cmd_stats)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)
