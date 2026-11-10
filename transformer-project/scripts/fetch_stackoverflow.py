"""
Build Stack Overflow question -> answer training documents (jsonl) from the
public posts Parquet files, without touching the Stack Overflow website or API.

Source of the data
  ClickHouse publishes the Stack Overflow "posts" table as one Parquet file per
  year (2008 .. April 2024), converted from the official Stack Exchange data
  dump:
    https://datasets-documentation.s3.eu-west-3.amazonaws.com/stackoverflow/parquet/posts/<year>.parquet
  Their docs describe it as CC BY-SA 4.0 data courtesy of Stack Overflow.

READ THIS BEFORE TRAINING ON IT
  * In July 2024 Stack Exchange moved the official dumps behind a login and,
    according to the Archive Team wiki, added terms that exclude LLM training
    for those downloads. This script uses the older public copy instead, but
    whether you may train on it is a licensing/terms question that is YOURS to
    decide - this is not legal advice.
  * The content itself is CC BY-SA (attribution + share-alike). Each output row
    keeps the question id and year so attribution can be reconstructed.
  * The data stops at April 2024: newer libraries/APIs will be missing.

Output: one jsonl line per Q&A pair:
  {"text": "<question>TITLE\\n\\nBODY<answer>ANSWER", "id": <question id>, "year": Y,
   "bucket": "python" | "adjacent"}   # adjacent = sql/regex/git/bash/algorithm... questions
<question> and <answer> are single special tokens in the project's tokenizer.

Pairing rule: the accepted answer if it passes the score filter, otherwise the
highest-scoring answer. Questions without a qualifying answer are dropped.

Usage:
  python scripts/fetch_stackoverflow.py --years 2008-2024 --out artifacts/stackoverflow.jsonl \\
      --workdir /content/so_parquet --min-q-score 1 --min-a-score 1
  # Python-tagged questions only (the default; Python 2 questions are dropped too).
  # Sample a fraction to control size:
  python scripts/fetch_stackoverflow.py ... --sample-rate 0.5
  # Python only, no SQL/regex/git/bash/algorithm extras:        --extra-tags
  # Every language (not what this project wants):               --tags --exclude-tags
  # Inspect the real column names first:
  python scripts/fetch_stackoverflow.py --years 2012 --probe
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

# Language-neutral topics that help a Python coder (SQL, regex, shell, git, algorithms...).
EXTRA_TAGS = [
    "sql", "mysql", "postgresql", "sqlite", "regex", "git", "github", "bash", "shell", "linux", "unix",
    "command-line", "awk", "sed", "algorithm", "data-structures", "sorting", "recursion",
    "dynamic-programming", "graph-algorithm", "big-o", "time-complexity", "binary-search",
    "linked-list", "binary-tree", "complexity-theory", "docker", "json", "csv",
]
# A question that has one of these tags is about ANOTHER language's ecosystem. It is kept only when
# it is also tagged python (so "sql + java" is dropped, "python + sql" is kept).
# Matching is on the tag's base name AND its "-variants": "ruby" also covers ruby-on-rails and
# ruby-on-rails-3, "java" covers java-8 / java-ee, "c++" covers c++11, "react" covers react-hooks...
OTHER_LANG_TAGS = [
    "java", "javascript", "c#", "c++", "c", "php", "ruby", "rails", "swift", "kotlin", "go", "golang", "rust",
    "typescript", "objective-c", "android", "ios", "xcode", "cocoa", "node.js", "nodejs", "npm", "express",
    "reactjs", "react", "angular", "angularjs", "vue.js", "vuejs", "jquery", "next.js", ".net", "asp.net",
    "html", "css", "perl", "scala", "r", "vba", "excel", "matlab", "vb.net", "vb6", "laravel", "symfony",
    "codeigniter", "cakephp", "yii", "magento", "drupal", "joomla", "wordpress", "spring", "hibernate", "jpa",
    "jdbc", "servlets", "jsp", "entity-framework", "linq", "ms-access", "unity3d", "unity", "visual-studio",
    "flutter", "dart", "lua", "haskell", "powershell", "delphi", "google-apps-script", "mysqli", "pdo",
    "activerecord", "sinatra", "grails", "groovy", "clojure", "elixir", "erlang", "fortran", "cobol",
]

URL = "https://datasets-documentation.s3.eu-west-3.amazonaws.com/stackoverflow/parquet/posts/{year}.parquet"

# ----------------------------------------------------------------------------
# HTML -> plain text with fenced code blocks
# ----------------------------------------------------------------------------
_PRE_CODE = re.compile(r"<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>", re.S | re.I)
_PRE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.S | re.I)
_CODE = re.compile(r"<code[^>]*>(.*?)</code>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")


def html_to_text(raw: str) -> str:
    blocks: list[str] = []

    def stash(m: re.Match) -> str:
        blocks.append(html.unescape(_TAG.sub("", m.group(1))).strip("\n"))
        return f"\n\x00{len(blocks) - 1}\x00\n"

    h = _PRE_CODE.sub(stash, raw)
    h = _PRE.sub(stash, h)
    h = _CODE.sub(lambda m: "`" + html.unescape(_TAG.sub("", m.group(1))) + "`", h)
    h = re.sub(r"<br\s*/?>", "\n", h, flags=re.I)
    h = re.sub(r"<li[^>]*>", "\n- ", h, flags=re.I)
    h = re.sub(r"</(p|h[1-6]|blockquote|ul|ol|tr|div)>", "\n\n", h, flags=re.I)
    h = html.unescape(_TAG.sub("", h))
    h = re.sub(r"\x00(\d+)\x00", lambda m: "```\n" + blocks[int(m.group(1))] + "\n```", h)
    h = re.sub(r"[ \t]+\n", "\n", h)
    return re.sub(r"\n{3,}", "\n\n", h).strip()


# ----------------------------------------------------------------------------
def download(year: int, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    dst = workdir / f"{year}.parquet"
    if dst.exists():
        return dst
    tmp = dst.with_suffix(".part")
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(URL.format(year=year), headers={"User-Agent": "transformer-project/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(8 * 1024 * 1024):
                    f.write(chunk)
            tmp.replace(dst)
            return dst
        except Exception as e:  # noqa: BLE001 - network errors are varied; retry then fail loudly
            print(f"  download {year} failed (attempt {attempt}/3): {e}", file=sys.stderr)
            time.sleep(5 * attempt)
    raise SystemExit(f"Could not download {URL.format(year=year)}")


def has_tag(tagset: set, names) -> bool:
    """True if any tag is a name or a variant of it: '-variants' (ruby -> ruby-on-rails-3,
    java -> java-8) and version suffixes (c++ -> c++11, swift -> swift3, angular -> angular2)."""
    for t in tagset:
        for n in names:
            if t == n:
                return True
            if t.startswith(n):
                rest = t[len(n):]
                if rest.startswith("-") or rest.lstrip("0123456789.") == "":
                    return True
    return False


def to_str(x) -> str:
    """The real parquet files store Title/Body/Tags as raw bytes; accept bytes, str or null."""
    if x is None or (isinstance(x, float) and x != x):
        return ""
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x).decode("utf-8", errors="replace")
    return str(x)


def post_type(series):
    """PostTypeId as int (1 question, 2 answer), whatever the parquet stores."""
    import pandas as pd

    if series.dtype == object:
        series = series.map(lambda v: {"Question": 1, "Answer": 2, "1": 1, "2": 2}.get(to_str(v), 0))
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def process_year(path: Path, year: int, args, out) -> tuple[int, int]:
    import pandas as pd
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    cols = set(pf.schema_arrow.names)
    need = {"Id", "PostTypeId", "ParentId", "AcceptedAnswerId", "Score", "Title", "Body", "Tags"}
    missing = need - cols
    if missing:
        raise SystemExit(f"{path.name}: missing columns {sorted(missing)}; actual: {sorted(cols)}")

    # ---- pass 1: small columns only -> decide which (question, answer) pairs to keep
    meta_cols = ["Id", "PostTypeId", "ParentId", "AcceptedAnswerId", "Score"] + (
        ["Tags"] if (args.tags or args.exclude_tags) else [])
    df = pf.read(columns=meta_cols).to_pandas()
    # The real files store some of these as text/object (ParentId in particular, with
    # empty values for questions), so force every key column to numbers before merging.
    for c in ("Id", "ParentId", "AcceptedAnswerId", "Score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Id"])
    df["Score"] = df["Score"].fillna(0)
    df["pt"] = post_type(df["PostTypeId"])
    q = df[(df.pt == 1) & (df.Score >= args.min_q_score)].copy()
    a = df[(df.pt == 2) & (df.Score >= args.min_a_score) & df.ParentId.notna()].copy()
    a["ParentId"] = a["ParentId"].astype("int64")
    q["bucket"] = "python"
    if args.tags or args.exclude_tags:
        tagsets = q.Tags.map(lambda t: set(re.findall(r"[^<>|]+", to_str(t).lower())))
        keep = pd.Series(True, index=q.index)
        if args.tags:
            want = {t.lower() for t in args.tags}
            extra = {t.lower() for t in args.extra_tags}
            other = {t.lower() for t in args.exclude_langs}
            is_py = tagsets.map(lambda ts: has_tag(ts, want))              # python, python-3.x, python-requests...
            is_adj = tagsets.map(lambda ts: has_tag(ts, extra) and not has_tag(ts, other))
            keep &= is_py | is_adj
            q["bucket"] = is_py.map({True: "python", False: "adjacent"})
        if args.exclude_tags:
            bad = {t.lower() for t in args.exclude_tags}
            keep &= ~tagsets.map(lambda ts: has_tag(ts, bad))
        q = q[keep]
    if args.sample_rate < 1.0:   # deterministic: depends only on the question id
        q = q[(q.Id.astype("int64") * 2654435761 % 1000) < int(args.sample_rate * 1000)]
    best = (a.sort_values(["ParentId", "Score"], ascending=[True, False])
              .drop_duplicates("ParentId")[["ParentId", "Id"]].rename(columns={"Id": "best_a"}))
    q["Id"] = q["Id"].astype("int64")
    q = q.merge(best, left_on="Id", right_on="ParentId", how="inner")
    acc_ok = q.AcceptedAnswerId.isin(set(a.Id))
    q["ans"] = q.best_a.where(~acc_ok, q.AcceptedAnswerId).astype("int64")
    pairs = dict(zip(q.Id.astype("int64"), q.ans))
    bucket_of = dict(zip(q.Id.astype("int64"), q.bucket))
    del df, a, best
    if not pairs:
        return 0, 0
    need_ids = set(pairs) | set(pairs.values())

    # ---- pass 2: bodies, in batches, keeping only what we need
    q_text: dict[int, str] = {}
    a_text: dict[int, str] = {}
    import pyarrow as pa
    import pyarrow.compute as pc

    value_set = pa.array(sorted(need_ids), type=pa.int64())
    for batch in pf.iter_batches(batch_size=50_000, columns=["Id", "Title", "Body"]):
        # filter in Arrow first: converting every Body to Python objects only to throw most away is slow
        ids = pc.cast(batch.column("Id"), pa.int64(), safe=False)
        batch = batch.filter(pc.is_in(ids, value_set=value_set))
        if batch.num_rows == 0:
            continue
        b = batch.to_pandas()
        b["Id"] = pd.to_numeric(b["Id"], errors="coerce")
        for rid, title, body in zip(b.Id, b.Title, b.Body):
            rid = int(rid)
            if rid in pairs:
                q_text[rid] = (to_str(title), to_str(body))
            else:
                a_text[rid] = to_str(body)

    n = chars = 0
    cap = args.max_mb_per_year * 1_000_000 if args.max_mb_per_year else None
    for qid, aid in pairs.items():
        if qid not in q_text or aid not in a_text:
            continue
        title, qbody = q_text[qid]
        qt, at = html_to_text(qbody), html_to_text(a_text[aid])
        if len(qt) < args.min_chars or len(at) < args.min_chars:
            continue
        doc = f"<question>{title.strip()}\n\n{qt}<answer>{at}"
        if len(doc) > args.max_chars:
            continue
        out.write(json.dumps({"text": doc, "id": qid, "year": year, "bucket": bucket_of[qid]},
                             ensure_ascii=False) + "\n")
        n += 1
        chars += len(doc)
        if cap and chars >= cap:
            break
    return n, chars


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="2008-2024", help="e.g. 2008-2024 or 2015,2016")
    ap.add_argument("--out", default="artifacts/stackoverflow.jsonl")
    ap.add_argument("--workdir", default="artifacts/so_parquet")
    ap.add_argument("--keep-parquet", action="store_true", help="do not delete each year's parquet after use")
    ap.add_argument("--min-q-score", type=int, default=1)
    ap.add_argument("--min-a-score", type=int, default=1)
    ap.add_argument("--tags", nargs="*", default=["python"],
                    help="keep only questions with any of these tags. DEFAULT: python (this project is a Python "
                         "model). Pass `--tags` with no values to keep every language.")
    ap.add_argument("--extra-tags", nargs="*", default=EXTRA_TAGS,
                    help="language-neutral topics kept IN ADDITION to --tags (sql, regex, git, bash, linux, "
                         "algorithms, ...), unless the question is also tagged with another language "
                         "(--exclude-langs). `--extra-tags` alone disables them.")
    ap.add_argument("--exclude-langs", nargs="*", default=OTHER_LANG_TAGS,
                    help="tags of other languages/ecosystems; applies only to the --extra-tags questions")
    ap.add_argument("--exclude-tags", nargs="*", default=["python-2.7", "python-2.x", "python-2"],
                    help="drop questions carrying any of these tags (default: Python 2). `--exclude-tags` alone disables")
    ap.add_argument("--sample-rate", type=float, default=1.0, help="deterministic fraction of questions to keep")
    ap.add_argument("--min-chars", type=int, default=30)
    ap.add_argument("--max-chars", type=int, default=24_000, help="skip pairs longer than this")
    ap.add_argument("--max-mb-per-year", type=int, default=0, help="optional safety cap on text per year (0 = none)")
    ap.add_argument("--probe", action="store_true", help="download one year, print columns + 1 converted example, exit")
    args = ap.parse_args()

    if "-" in args.years:
        lo, hi = map(int, args.years.split("-"))
        years = list(range(lo, hi + 1))
    else:
        years = [int(y) for y in args.years.split(",")]

    if args.probe:
        import pyarrow.parquet as pq
        path = download(years[0], Path(args.workdir))
        pf = pq.ParquetFile(path)
        print("columns:", pf.schema_arrow.names, "| rows:", pf.metadata.num_rows)
        import io
        buf = io.StringIO()
        n, _ = process_year(path, years[0], argparse.Namespace(**{**vars(args), "sample_rate": min(args.sample_rate, 0.002)}), buf)
        first = buf.getvalue().splitlines()[:1]
        print(f"pairs from a 0.2% sample: {n}")
        if first:
            print(json.loads(first[0])["text"][:800])
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    total_n = total_chars = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for y in years:
            t0 = time.time()
            path = download(y, Path(args.workdir))
            n, chars = process_year(path, y, args, out)
            out.flush()
            total_n += n
            total_chars += chars
            if not args.keep_parquet:
                path.unlink(missing_ok=True)
            print(f"{y}: {n:,} pairs, {chars/1e6:,.0f} MB text ({time.time()-t0:.0f}s) | "
                  f"total {total_n:,} pairs, ~{total_chars/4/1e9:.2f}B tokens (4 chars/token)", flush=True)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
