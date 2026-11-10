"""
Repo-level pretraining data formatter.

Turns a cloned repository into training sequences that preserve
cross-file structure, instead of flat isolated files.

Usage:
    python repo_concat.py /path/to/cloned/repo --out repo_sequences.jsonl
"""

import os
import json
import argparse
from pathlib import Path

# Extensions worth including — extend as needed for your target languages
CODE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".h"}

# Directories/patterns that are noise, not signal
SKIP_DIRS = {
    "node_modules", "vendor", ".git", "dist", "build",
    "__pycache__", ".venv", "venv", "target", "bin", "obj",
}
SKIP_FILE_PATTERNS = (".min.js", ".lock", ".generated.", ".pb.go")

MAX_SEQUENCE_TOKENS_APPROX = 8000   # rough chars-as-tokens budget per sequence; tune to your context window
CHARS_PER_TOKEN_ESTIMATE = 4        # rough heuristic for English/code text


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    if any(pat in path.name for pat in SKIP_FILE_PATTERNS):
        return True
    return path.suffix not in CODE_EXTENSIONS


def collect_files(repo_root: Path):
    """Walk repo, keep real source files, skip noise."""
    files = []
    for p in repo_root.rglob("*"):
        if p.is_file() and not should_skip(p):
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if len(content.strip()) < 20:
                continue  # skip near-empty files
            files.append((p.relative_to(repo_root), content))
    return files


def order_files_by_directory(files):
    """
    Simple dependency-aware-ish ordering: group by directory depth/path,
    so related modules end up adjacent. (For true import-graph ordering,
    parse imports per language and topologically sort instead.)
    """
    return sorted(files, key=lambda x: (len(x[0].parts), str(x[0])))


def build_sequences(files, repo_name: str, repo_name_probability: float = 0.5):
    """Pack ordered files into training sequences under a rough token budget.

    Uses the <repo_name>/<file_sep> convention StarCoder2 and Qwen2.5-Coder
    train on (verified via their public tokenizer configs and the StarCoder2
    paper), rather than an ad-hoc <file_path> tag — this way the tokens
    trained on here match a scheme the tokenizer/model can be built around
    deliberately (e.g. added as explicit special tokens so they don't get
    fragmented into sub-word pieces).

    repo_name_probability: StarCoder2 prepends the repo name only ~50% of
    the time, specifically so the model also learns to work without that
    metadata present (e.g. at inference time you may not always know/want
    to supply a repo name). Set to 1.0 to always include it, 0.0 to never.
    """
    import random
    sequences = []
    current_parts = []
    current_len = 0
    include_repo_name = random.random() < repo_name_probability

    for idx, (rel_path, content) in enumerate(files):
        prefix = f"<repo_name>{repo_name}" if (idx == 0 and include_repo_name) else ""
        block = f"{prefix}<file_sep>{rel_path.as_posix()}\n{content}\n\n"
        block_len_est = len(block) / CHARS_PER_TOKEN_ESTIMATE

        if current_len + block_len_est > MAX_SEQUENCE_TOKENS_APPROX and current_parts:
            sequences.append("".join(current_parts))
            current_parts = []
            current_len = 0

        current_parts.append(block)
        current_len += block_len_est

    if current_parts:
        sequences.append("".join(current_parts))

    return sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_path", help="Path to a cloned repository")
    ap.add_argument("--out", default="repo_sequences.jsonl")
    args = ap.parse_args()

    repo_root = Path(args.repo_path)
    repo_name = repo_root.name

    files = collect_files(repo_root)
    files = order_files_by_directory(files)
    sequences = build_sequences(files, repo_name)

    with open(args.out, "a", encoding="utf-8") as f:
        for seq in sequences:
            f.write(json.dumps({"repo": repo_name, "text": seq}) + "\n")

    print(f"{repo_name}: {len(files)} files -> {len(sequences)} training sequences written to {args.out}")


if __name__ == "__main__":
    main()
