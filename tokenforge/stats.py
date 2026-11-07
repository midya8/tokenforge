"""Corpus statistics for a trained tokenizer."""

import json
from collections import Counter


def _escape_md(text):
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`")


def _display(piece_bytes):
    text = piece_bytes.decode("utf-8", errors="replace")
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return text


def corpus_stats(tokenizer, text, top=10):
    """Tokenize text and compute compression and usage statistics."""
    ids = tokenizer.encode(text)
    byte_len = len(text.encode("utf-8"))
    token_count = len(ids)
    counts = Counter(ids)
    top_tokens = [
        {
            "id": tid,
            "count": count,
            "text": _display(tokenizer.token_bytes(tid)),
        }
        for tid, count in counts.most_common(top)
    ]
    return {
        "bytes": byte_len,
        "tokens": token_count,
        "unique_tokens": len(counts),
        "vocab_size": tokenizer.vocab_size,
        "compression_ratio": (byte_len / token_count) if token_count else 0.0,
        "bytes_per_token": (byte_len / token_count) if token_count else 0.0,
        "top_tokens": top_tokens,
    }


def render_terminal(stats):
    lines = [
        "bytes        %d" % stats["bytes"],
        "tokens       %d" % stats["tokens"],
        "unique       %d" % stats["unique_tokens"],
        "vocab size   %d" % stats["vocab_size"],
        "compression  %.4f bytes/token" % stats["bytes_per_token"],
    ]
    if stats["top_tokens"]:
        lines.append("top tokens")
        for entry in stats["top_tokens"]:
            lines.append(
                "  %5d x%-4d %s" % (entry["id"], entry["count"], _escape_display(entry["text"]))
            )
    return "\n".join(lines)


def _escape_display(text):
    return text if text.strip() else repr(text)


def render_markdown(stats):
    lines = [
        "# TokenForge corpus statistics",
        "",
        "| metric | value |",
        "|---|---|",
        "| bytes | %d |" % stats["bytes"],
        "| tokens | %d |" % stats["tokens"],
        "| unique tokens | %d |" % stats["unique_tokens"],
        "| vocab size | %d |" % stats["vocab_size"],
        "| bytes per token | %.4f |" % stats["bytes_per_token"],
        "",
    ]
    if stats["top_tokens"]:
        lines += [
            "## Most frequent tokens",
            "",
            "| token id | count | text |",
            "|---|---|---|",
        ]
        for entry in stats["top_tokens"]:
            lines.append(
                "| %d | %d | `%s` |"
                % (entry["id"], entry["count"], _escape_md(entry["text"]))
            )
        lines.append("")
    return "\n".join(lines)


def write_json_report(stats, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
        fh.write("\n")
