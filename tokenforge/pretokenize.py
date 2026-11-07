"""Regex pre-tokenization for byte-level BPE.

The pattern is a simplified, ASCII-oriented version of the GPT-2 split:
contractions, words, numbers, runs of punctuation/symbols and whitespace
are split so that merges never cross these boundaries. Keeping merges
inside pre-tokens makes training faster and encoding deterministic.
"""

import re

# Order matters: the regex engine tries alternatives left to right.
PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"      # contractions: 's 'd 'm 't 'll 've 're
    r"| ?[A-Za-z]+"              # words, optionally led by one space
    r"| ?[0-9]+"                 # numbers, optionally led by one space
    r"| ?[^\sA-Za-z0-9]+"        # symbol runs, optionally led by one space
    r"|\s+(?!\S)"                # trailing whitespace
    r"|\s+"                      # any remaining whitespace
)


def pretokenize(text):
    """Split text into pre-token strings."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return PATTERN.findall(text)
