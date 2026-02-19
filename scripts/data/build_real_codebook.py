#!/usr/bin/env python3
"""Build an authentic Dynamic Codebook from a training corpus.

This replaces the synthetic generator by actually parsing text,
encoding it into topological signatures via the sep_text_manifold,
and counting true token frequencies to populate the dictionary.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Use the established Manifold API
from src.manifold.sidecar import encode_text


def build_authentic_codebook(
    corpus_path: Path, vocab_path: Path, output_path: Path, max_bytes: int = 10_000_000
):
    try:
        with open(vocab_path) as f:
            vocab_data = json.load(f)
            # signatures might not be perfectly matched, but we can try
            signatures = set(vocab_data["signatures"])
    except FileNotFoundError:
        print(f"Vocab {vocab_path} not found. Cannot filter codebook.")
        return

    print(f"Reading corpus {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read(max_bytes)

    print(f"Tokenizing entire corpus to establish global spatial index...")
    token_matches = list(re.finditer(r"\b\w+\b", text))
    token_starts = [m.start() for m in token_matches]
    global_position = len(token_matches)
    print(f"Corpus contains {global_position} tokens.")

    print(f"Encoding {len(text)} characters into manifold space...")
    encode_result = encode_text(
        text, window_bytes=512, stride_bytes=384, precision=3, use_native=False
    )
    windows = encode_result.windows

    print(f"Generated {len(windows)} signatures.")

    sig_metadata = defaultdict(lambda: {"token_counts": Counter(), "positions": []})

    import bisect

    for w in windows:
        if w.signature not in signatures:
            continue

        center_char = w.char_start + (w.char_end - w.char_start) // 2
        idx = bisect.bisect_left(token_starts, center_char)

        if idx < global_position:
            match = token_matches[idx]
            token_str = match.group(0).lower()
            if len(token_str) > 2:
                sig_metadata[w.signature]["token_counts"][token_str] += 1
                sig_metadata[w.signature]["positions"].append(idx)

    codebook = {
        "window_size": 512,
        "decay_factor": 0.95,
        "global_position": global_position,
        "entries": {},
        "spatial_index": {},
    }

    print(
        f"Populating codebook with top tokens for {len(sig_metadata)} unique signatures..."
    )

    sig_list = []

    # Assign tokens to signatures based on frequency
    for sig, meta in sig_metadata.items():
        if not meta["token_counts"]:
            continue
        top_tokens = [token for token, count in meta["token_counts"].most_common(5)]

        positions = meta["positions"]
        last_seen = max(positions) if positions else 0

        codebook["entries"][sig] = {
            "signature": sig,
            "tokens": top_tokens,
            "positions": sorted(list(set(positions))),
            "frequency": float(sum(meta["token_counts"].values())),
            "last_seen": last_seen,
        }
        sig_list.append(sig)

    # Build spatial index
    for i, sig in enumerate(sig_list):
        neighbors = []
        for j in range(max(0, i - 5), min(len(sig_list), i + 6)):
            if i != j:
                neighbors.append(sig_list[j])
        codebook["spatial_index"][sig] = neighbors

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(codebook, f, indent=2)

    print(
        f"Authentic codebook created with {len(codebook['entries'])} entries at {output_path}"
    )


if __name__ == "__main__":
    corpus = Path("wikitext-103/wiki.train.tokens")
    vocab = Path("output/production/manifold_dataset/vocab.json")
    out = Path("output/production/authentic_codebook.json")
    if not corpus.exists():
        print(f"Corpus {corpus} not found. Ensure wikitext-103 is downloaded.")
        sys.exit(1)
    build_authentic_codebook(corpus, vocab, out)
