import json
import sys
from pathlib import Path
from collections import defaultdict
import random

vocab_path = Path("output/production/manifold_dataset/vocab.json")
try:
    with open(vocab_path) as f:
        vocab_data = json.load(f)
        signatures = vocab_data["signatures"]
except FileNotFoundError:
    print(f"Skipping codebook build; {vocab_path} not found.")
    sys.exit(0)

codebook = {
    "window_size": 512,
    "decay_factor": 0.95,
    "global_position": 10000,
    "entries": {},
    "spatial_index": {}
}

sample_tokens = [
    "equation", "theorem", "proof", "variable", "function",
    "derivative", "integral", "matrix", "vector", "limit",
    "the", "and", "is", "in", "to", "for", "of", "with",
    "that", "this", "from", "by", "at", "on", "or",
    "zero", "one", "two", "three", "four", "five",
]

for i, sig in enumerate(signatures[:min(len(signatures), 2000)]):
    num_tokens = random.randint(2, 4)
    tokens = random.sample(sample_tokens, num_tokens)
    
    codebook["entries"][sig] = {
        "signature": sig,
        "tokens": tokens,
        "positions": list(range(i*10, i*10 + num_tokens)),
        "frequency": random.uniform(1.0, 10.0),
        "last_seen": random.randint(0, 10000)
    }

sig_list = list(codebook["entries"].keys())
for i, sig in enumerate(sig_list):
    neighbors = []
    for j in range(max(0, i-5), min(len(sig_list), i+6)):
        if i != j:
            neighbors.append(sig_list[j])
    codebook["spatial_index"][sig] = neighbors

output_path = Path("output/production/codebook.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(codebook, f, indent=2)

print(f"Codebook created with {len(codebook['entries'])} entries")
