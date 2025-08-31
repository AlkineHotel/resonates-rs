#!/usr/bin/env python3
import sys, json, argparse, torch
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

def add_prompt(texts: List[str], style: str) -> List[str]:
    if not style:
        return texts
    style = style.lower()
    if style in ("e5", "bge", "passage"):
        prefix = "passage: "
        return [prefix + t for t in texts]
    return texts

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.add_argument("--style", default="bge", help="prompt style: bge/e5/passage/''")
    args = p.parse_args()

    torch.set_num_threads(max(torch.get_num_threads(), 4))
    model = SentenceTransformer(args.model, device=args.device)
    # Ensure truncation to max_length if the model supports setting it
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = args.max_length

    payload = sys.stdin.read()
    items: List[Dict] = json.loads(payload)
    ids = [int(x["id"]) for x in items]
    texts = [x["text"] for x in items]
    texts = add_prompt(texts, args.style)

    vecs = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=args.normalize,
        show_progress_bar=False,
    )
    # Safety: handle any NaNs/Infs
    vecs = np.nan_to_num(vecs, copy=False)

    out = [{"id": i, "vec": v.tolist()} for i, v in zip(ids, vecs)]
    sys.stdout.write(json.dumps(out))

if __name__ == "__main__":
    main()