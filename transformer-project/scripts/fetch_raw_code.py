"""
Fetch individual Java files (no repo grouping) for the "raw code" 70%
slice of the data mix. Uses GitHub's code search to sample files across
many different repos, rather than deeply cloning any single one — this is
deliberately the flat/file-level counterpart to repo_concat.py's
repo-grouped output.

Files are separated with a bare <file_sep> tag (no <repo_name>) so the
model still gets an explicit file-boundary signal without implying any
repo-level relationship between adjacent files, since there isn't one.

Usage:
    python fetch_raw_code.py --language Java --out data/raw_code --max-files 2000
"""
from __future__ import annotations

import argparse
import os
import time

import requests


def search_files(language: str, token: str, max_files: int):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    collected = []
    page = 1
    while len(collected) < max_files:
        # Code search API: broad query, filtered by language and a permissive-license signal
        params = {
            "q": f"language:{language} stars:>50",
            "sort": "indexed",
            "per_page": 100,
            "page": page,
        }
        r = requests.get("https://api.github.com/search/repositories", params=params, headers=headers)
        if r.status_code == 403:
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            time.sleep(max(reset - time.time(), 1))
            continue
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            break
        for repo in items:
            collected.append(repo["full_name"])
            if len(collected) >= max_files:
                break
        page += 1
        time.sleep(1)
    return collected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="Java")
    ap.add_argument("--out", default="data/raw_code")
    ap.add_argument("--max-files", type=int, default=2000)
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("Set GITHUB_TOKEN in your environment first.")

    os.makedirs(args.out, exist_ok=True)
    repos = search_files(args.language, token, args.max_files)
    print(f"Found {len(repos)} candidate repos — clone a sample of these with your "
          f"existing clone_repos.py, then take ONE representative file per repo "
          f"(not the whole repo) to keep this genuinely file-level rather than repo-level.")

    with open(os.path.join(args.out, "repo_list.txt"), "w") as f:
        f.write("\n".join(repos))
    print(f"Repo list written to {args.out}/repo_list.txt")


if __name__ == "__main__":
    main()
