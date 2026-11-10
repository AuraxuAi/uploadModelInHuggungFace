"""
Extract (signature, body) pairs from real Python source files using the ast
module, to build an SFT dataset that directly targets the gap observed in
pretraining-only generation: the model reproduces plausible file-level
boilerplate (imports, classes) but doesn't continue a given function body
coherently based on its actual signature/name.

Each extracted example becomes one SFT training pair:
    prompt   = the function signature + docstring (if present)
    response = the function body

Usage (after Step 1-5's the-stack-dedup fetch already gave you real Python
source — reuse the SAME raw text, don't re-fetch):
    python extract_sft_pairs.py --source data/all_python_code.txt --out data/sft_pairs.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json


def extract_function_pairs(source_code: str, min_body_lines: int = 2, max_body_chars: int = 2000):
    """Parse one Python file's source and yield (signature, body) pairs for
    each top-level or nested function definition with a real, non-trivial
    body (skips one-liners like `def f(): pass`, which teach nothing useful
    here and would dilute the signal)."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return  # skip files that don't parse cleanly — common in scraped corpora

    lines = source_code.splitlines()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not hasattr(node, "end_lineno") or node.end_lineno is None:
            continue

        # Reconstruct the exact source signature line(s), not a re-serialized
        # approximation — real formatting/spacing is part of what the model
        # should learn to expect as "prompt".
        sig_end = node.body[0].lineno - 1 if node.body else node.lineno
        signature_lines = lines[node.lineno - 1: sig_end]
        signature = "\n".join(signature_lines)

        body_lines = lines[sig_end: node.end_lineno]
        body = "\n".join(body_lines)

        if len(body_lines) < min_body_lines or len(body) > max_body_chars:
            continue
        if not body.strip():
            continue

        yield signature, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                     help="Text file containing Python source, files separated by <file_sep> "
                          "(matches the format combine_java_files.py / the-stack fetch already produce)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pairs", type=int, default=None)
    args = ap.parse_args()

    with open(args.source, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    files = content.split("<file_sep>")
    pair_count = 0

    with open(args.out, "w", encoding="utf-8") as out_f:
        for file_content in files:
            if not file_content.strip():
                continue
            # Strip the leading "path\n" line that combine scripts prepend after <file_sep>
            file_body = file_content.split("\n", 1)[1] if "\n" in file_content else file_content

            for signature, body in extract_function_pairs(file_body):
                out_f.write(json.dumps({"prompt": signature, "response": body}) + "\n")
                pair_count += 1
                if args.max_pairs and pair_count >= args.max_pairs:
                    print(f"Reached max_pairs={args.max_pairs}, stopping.")
                    print(f"Extracted {pair_count:,} (signature, body) pairs -> {args.out}")
                    return

    print(f"Extracted {pair_count:,} (signature, body) pairs -> {args.out}")


if __name__ == "__main__":
    main()
