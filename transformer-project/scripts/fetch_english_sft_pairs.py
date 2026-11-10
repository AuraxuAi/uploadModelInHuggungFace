"""
Fetch English instruction/response pairs for SFT, using the SAME jsonl
schema (prompt/response) as extract_sft_pairs.py's code pairs — no changes
needed to SFTPairDataset, cli.py, or the trainer; this just produces another
compatible data source.

Uses databricks/databricks-dolly-15k: real, human-written instruction data
(not model-generated), permissively licensed (CC-BY-SA-3.0), confirmed
fields: instruction, context, response, category.
    https://huggingface.co/datasets/databricks/databricks-dolly-15k

Usage:
    python fetch_english_sft_pairs.py --out data/english_sft_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/english_sft_pairs.jsonl")
    ap.add_argument("--max-pairs", type=int, default=None)
    args = ap.parse_args()

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    count = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for row in ds:
            instruction = row["instruction"].strip()
            context = row["context"].strip()
            response = row["response"].strip()

            if not instruction or not response:
                continue

            # Fold context into the prompt when present, matching the
            # natural way a person would actually phrase it — this keeps
            # the "prompt" side self-contained, same idea as the code SFT
            # pairs' signature-as-prompt.
            if context:
                prompt = f"{instruction}\n\nContext: {context}"
            else:
                prompt = instruction

            out_f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
            count += 1
            if args.max_pairs and count >= args.max_pairs:
                break

    print(f"Wrote {count:,} English instruction pairs -> {args.out}")


if __name__ == "__main__":
    main()
