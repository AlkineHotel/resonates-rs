# code-analyzer (AST chunking + cross-file similarity)

This CLI splits code by AST subtrees and finds similar chunks across files to help you:
- Eliminate redundancy/near-duplicates
- Surface “misplaced” code (similar logic living in the wrong module)
- Triage large refactors safely

It is non-LLM by default (token-based SimHash + Jaccard). Optionally, you can plug in a BERT-class embedding model (e.g., CodeBERT) for semantic matches.

## Install

Requirements:
- Rust 1.70+ (for building)
- For embedding mode only: Python 3.9+ and `sentence-transformers`

Build:
```bash
cargo build --release
```

## Quick start

Token-based (fast, no embeddings):
```bash
# Analyze current repo, find near-duplicates across files, save JSON
target/release/code-analyzer \
  --path . \
  --similarity token \
  --sim-threshold 0.85 \
  --sim-top-k 500 \
  --sim-cross-file-only true \
  --sim-include-snippets \
  --output analysis.json \
  --sim-output similarity.json
```

Embedding-based (optional semantic layer, needs Python):
```bash
python3 -m pip install -U sentence-transformers
target/release/code-analyzer \
  --path . \
  --similarity embedding \
  --embedder-cmd "python3 embed.py" \
  --sim-threshold 0.80 \
  --sim-top-k 300 \
  --sim-include-snippets
```

Outputs:
- `analysis.json`: AST chunks per file (ranges, sizes, types)
- `similarity.json`: list of high-confidence similar chunk pairs across files

## What’s happening under the hood

### AST chunking

We use tree-sitter via `code_splitter` to cut files into structural chunks (functions/methods/classes/blocks) within a size budget (`--max-size`), which improves the signal for similarity.

Nuances:
- Smaller `--max-size` tends to create more, tighter chunks; this increases recall (more opportunities to match) but can increase false positives.
- Very large chunks mix responsibilities and dilute similarity signals.
- Language autodetection is by extension (.rs/.ts/.tsx/.js/.jsx). You can override via `--language`.

### Token-based similarity (default)

This path is entirely non-LLM and robust for exact/near-exact clones and light edits.

1) Identifier tokenization
- We extract identifier-like tokens with a light regex, split on snake_case and camelCase, and drop common keywords (stop list).
- Result is a set of tokens per chunk (we use sets for speed and robustness to repetition).
- `--sim-min-tokens` drops very small chunks that aren’t informative.

2) Candidate generation with SimHash (LSH-ish)
- SimHash compresses a token set into a 64-bit fingerprint that preserves similarity (identical or very similar sets produce similar hashes).
- We do “banding”: split the 64 bits into bands of `--sim-band-bits` and only compare pairs that share at least one band. This prunes the O(n²) search.

Trade-offs with `--sim-band-bits`:
- Smaller bands (e.g., 8) → more candidate pairs = higher recall, higher runtime, more false positives to filter.
- Larger bands (e.g., 16) → fewer candidates = lower recall, faster.

3) Scoring with Jaccard similarity
- Jaccard = |intersection(tokensA, tokensB)| / |union(tokensA, tokensB)|
- Captures set overlap; insensitive to order and repetition.
- Good for near-duplicates that differ by minor edits or variable renames (since splits/identifiers dominate signal).

Threshold guidance (`--sim-threshold`):
- 0.90–0.95: very high confidence duplicates (Type-1/2)
- 0.80–0.89: near-miss with small edits (Type-2/3)
- 0.70–0.79: broader “same idea” but more noise; review needed

Nuances, caveats:
- Using sets means repetition frequency doesn’t influence the score; this reduces false positives from repeated boilerplate but can miss “pattern frequency” signals. If you need frequency, switch to multiset Jaccard in a future extension.
- Comments may leak tokens (e.g., license headers). You can mitigate by:
  - Running with `--sim-cross-file-only` (skips same-file headers)
  - Raising `--sim-min-tokens` beyond header size
  - Adding more stopwords (extendable in code)
- SimHash is only a candidate filter; final decision is Jaccard over real tokens.

### Embedding-based similarity (optional)

You can plug in an external process for embeddings (e.g., CodeBERT). We feed chunk snippets to the command and expect vectors back, then compute cosine similarity.

- Cosine similarity: (A·B)/(|A||B|), higher is more similar.
- Typical thresholds:
  - 0.85–0.92: very similar code logic
  - 0.75–0.84: conceptually similar but with structural differences
- Better at surfacing “same intent” when tokens diverge (different naming). Slower and costlier; use `--sim-top-k` to cap output.

Reference script: `embed.py` (expects stdin JSON with `[{id, text}]`, returns `[{id, vec}]`).

### Detecting “misplaced code”

- Turn on `--sim-cross-file-only` to focus on cross-file issues.
- The report includes `common_path_prefix` (number of shared path segments). Smaller values often indicate cross-module duplication or misplaced logic (e.g., `ui/` vs `domain/`).
- Combine with an architecture linter for enforcement (e.g., ArchUnit/import-linter) if needed.

## CLI options (most relevant)

- `--similarity [none|token|embedding]`
- `--sim-threshold <f32>`: Jaccard (token) or cosine (embedding)
- `--sim-top-k <usize>`: cap pairs in `similarity.json` for triage
- `--sim-band-bits <usize>`: SimHash banding; lower = more candidates
- `--sim-min-tokens <usize>`: ignore tiny chunks
- `--sim-cross-file-only [true|false]`: ignore same-file matches
- `--sim-include-snippets`: include code text in similarity results
- `--embedder-cmd "<cmd>"`: external embedder command for embedding mode
- `--max-size <usize>`: AST chunk size (chars)

## JSON outputs

`analysis.json` (per file):
```json
{
  "total_files": 2,
  "total_chunks": 12,
  "max_chunk_size": 2048,
  "files": [
    {
      "file_path": "src/foo.rs",
      "chunks": [
        {
          "subtree_description": "function_item",
          "start_line": 10,
          "end_line": 42,
          "start_byte": 123,
          "end_byte": 999,
          "size": 876
        }
      ]
    }
  ]
}
```

`similarity.json` (pairs):
```json
{
  "method": "token",
  "threshold": 0.85,
  "top_k": 500,
  "total_chunks": 120,
  "total_pairs": 37,
  "pairs": [
    {
      "score": 0.92,
      "method": "token",
      "overlap_tokens": 48,
      "union_tokens": 52,
      "same_file": false,
      "common_path_prefix": 1,
      "a": {
        "file_path": "src/domain/user.rs",
        "subtree_description": "function_item",
        "start_line": 40,
        "end_line": 88,
        "size": 1320,
        "snippet": "fn compute_roles(...) { ... }"
      },
      "b": {
        "file_path": "src/ui/user_view.rs",
        "subtree_description": "function_item",
        "start_line": 12,
        "end_line": 59,
        "size": 1298,
        "snippet": "pub fn get_roles(...) { ... }"
      }
    }
  ]
}
```

## Tuning guide

- Too many pairs? Increase `--sim-threshold`, increase `--sim-band-bits`, raise `--sim-min-tokens`, or lower `--sim-top-k`.
- Missing obvious clones? Lower `--sim-threshold` a bit (e.g., 0.80), decrease `--sim-band-bits` (e.g., 8 → 6), or reduce `--max-size` so functions aren’t merged into large chunks.
- Repeated license/header noise? Raise `--sim-min-tokens` and/or extend stopwords in `similarity.rs`. You can also preexclude paths via `--exclude`.

## Troubleshooting

- “Snippets are missing in similarity.json”
  - Ensure you pass `--sim-include-snippets`
  - Update to the latest binary with the snippet fix for token mode (this repo)
- “Embedding mode fails”
  - Make sure `--embedder-cmd` points to a command available in `sh -lc "..."`
  - The command must read JSON from stdin and write JSON to stdout as documented.
- “Very slow on big repos”
  - Use token mode first; if needed, run embedding mode only on suspected directories.
  - Increase `--sim-band-bits` to prune candidates aggressively.
  - Use `--sim-top-k` to reduce output volume.

## Notes on privacy and determinism

- Token mode is deterministic and offline.
- Embedding mode calls your local process only; no network calls unless your embedder does so.
- For stable results across runs, keep chunking parameters and stopwords fixed.

---

Appendix: concise definitions

- Jaccard similarity: size of intersection divided by size of union of two sets. Range [0,1]. Robust to reordering and duplicates; great for near-duplicate detection over token sets.
- SimHash: a locality-sensitive hashing scheme that maps high-dimensional features to a bit fingerprint where similar inputs yield similar bit patterns. We use it to quickly shortlist candidate pairs before exact scoring.
- Cosine similarity: dot product of normalized vectors; measures orientation (not magnitude). Used with embeddings to measure semantic closeness.