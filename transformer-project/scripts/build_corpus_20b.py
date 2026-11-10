"""
Build the ~20B-token, multi-source pretraining corpus (one tokenized .bin per
source), ready to be blended by scripts/build_data_mix.py.

Subcommands (run in this order):

  plan      Print the source table and the exact build_data_mix.py command.
  probe     Stream ONE example from every source and print its fields + the
            formatted text. RUN THIS FIRST: dataset names/configs/schemas on
            the Hub change, and it costs seconds instead of a day of wasted
            tokenization.
  sample    Write a small text sample (default ~1 GB total, proportional to the
            mix) for training the tokenizer on the NEW mix.
  tokenize  Stream every source, tokenize into resumable 1B-token shards
            (<out>/<name>/shard_*.bin, uint16) until each source hits its budget.
  merge     Concatenate each source's shards into <out>/<name>.bin.

Typical flow:
  python scripts/build_corpus_20b.py plan
  python scripts/build_corpus_20b.py probe
  python scripts/build_corpus_20b.py sample --out artifacts/tok_sample
  tp-train-tokenizer --files artifacts/tok_sample/*.txt --vocab-size 32000 \
      --out artifacts/tokenizer_20b_32k.json
  python scripts/build_corpus_20b.py tokenize --tokenizer artifacts/tokenizer_20b_32k.json \
      --out artifacts/src_20b
  python scripts/build_data_mix.py ...   # command printed by `plan`

Notes
  * Tokens are written as uint16 (needs vocab_size <= 65535): 20B tokens = 40 GB
    instead of 80 GB. The mix step and the training config must use the same dtype.
  * The tokenizer's post-processor already wraps every document in [BOS] ... [EOS],
    so no extra EOS is appended here (the older pipeline appended a second one).
  * `tokenize` is resumable: re-running it continues after the last finished shard
    (with --sync-dir pointing at Drive it survives a wiped Colab runtime).
  * Licensing: the-stack-dedup is permissively-licensed code, but you are still
    responsible for checking the terms of every dataset below before training on
    it or releasing a model. This script does not filter licenses.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

TOTAL_TOKENS = 20_000_000_000


# ----------------------------------------------------------------------------
# Formatters: turn one raw dataset row into one training document (a string).
# Structural tokens (<file_sep>, <commit_before>, ...) are special tokens in
# transformer_project.data.tokenizer.SPECIAL_TOKENS.
# ----------------------------------------------------------------------------
def fmt_text(row: dict) -> str:
    return row["text"]


def fmt_code_file(row: dict) -> str:
    return "<file_sep>" + row["content"]


MAX_SOLUTION_CHARS = 6000   # skip pathological competitive-programming solutions
PY3 = (3, "PYTHON3")        # CodeContests language enum: 1=PYTHON(py2) 2=CPP 3=PYTHON3 4=JAVA


def fmt_code_contests(row: dict) -> Optional[str]:
    """deepmind/code_contests: problem statement + up to 4 CORRECT Python 3 solutions."""
    sols = row.get("solutions") or {}
    py = [c for lang, c in zip(sols.get("language", []), sols.get("solution", []))
          if lang in PY3 and c.strip() and len(c) <= MAX_SOLUTION_CHARS][:4]
    if not py or not (row.get("description") or "").strip():
        return None
    return f"<question>{row['description'].strip()}" + "".join(f"<answer>{c.strip()}" for c in py)


def fmt_apps(row: dict) -> Optional[str]:
    """codeparrot/apps: problem + up to 3 solutions (stored as a JSON string)."""
    try:
        sols = json.loads(row.get("solutions") or "[]")
    except (TypeError, ValueError):
        return None
    sols = [c for c in sols if isinstance(c, str) and c.strip() and len(c) <= MAX_SOLUTION_CHARS][:3]
    if not sols or not (row.get("question") or "").strip():
        return None
    start = (row.get("starter_code") or "").strip()
    q = row["question"].strip() + (f"\n\nStarter code:\n{start}" if start else "")
    return f"<question>{q}" + "".join(f"<answer>{c.strip()}" for c in sols)


MAX_COMMIT_CHARS = 4000  # truncate huge file diffs (same limit as the 5B notebook)


def fmt_commit(row: dict) -> Optional[str]:
    old = (row.get("old_contents") or "")[:MAX_COMMIT_CHARS]
    new = (row.get("new_contents") or "")[:MAX_COMMIT_CHARS]
    msg = (row.get("message") or row.get("subject") or "").strip()
    if len(msg) < 5 or not new.strip():
        return None
    return f"<commit_before>{old}<commit_message>{msg}<commit_after>{new}"


def keep_code_file(row: dict) -> bool:
    """Quality filter for bigcode/the-stack-dedup (same rules as the 5B notebook)."""
    if row.get("max_line_length", 0) > 300 or row.get("alphanum_fraction", 1.0) < 0.25:
        return False
    if row.get("size", 10**9) < 50:
        return False
    head = row["content"]
    return not (head.strip().startswith("# Auto-generated") or "DO NOT EDIT" in head[:200])


def fmt_issue(row: dict) -> str:
    # Schema differs between releases; handle the common shapes, else fail loudly
    # so `probe` tells you what to change instead of silently training on junk.
    if isinstance(row.get("events"), list):
        parts = []
        for ev in row["events"]:
            body = ev.get("text") or ev.get("content") or ""
            if body:
                parts.append(body)
        if parts:
            return "<issue_start>" + "<issue_comment>".join(parts)
    for key in ("content", "text"):
        if row.get(key):
            return "<issue_start>" + row[key]
    raise KeyError(f"Unrecognised issues schema; fields: {sorted(row)}. Edit fmt_issue().")


@dataclass
class Source:
    name: str
    tokens: int                      # token budget (the "mix" size for this source)
    fmt: Callable[[dict], str]
    hf_name: Optional[str] = None
    hf_config: Optional[str] = None
    hf_data_dir: Optional[str] = None
    hf_revision: Optional[str] = None
    keep: Optional[Callable[[dict], bool]] = None   # row filter
    # Memory controls for streaming. Rows of some datasets are huge (CodeContests: thousands of
    # solutions + tests per row), so the shuffle buffer and the read batch must be tiny for them.
    shuffle_buffer: int = 10_000
    columns: Optional[list] = None      # read only these columns (parquet-backed datasets)
    batch_size: Optional[int] = None    # rows per read batch (parquet-backed datasets)
    split: str = "train"
    needs_local: bool = False        # content not on the Hub -> supply via --local name=path.jsonl
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        return self.tokens / TOTAL_TOKENS


B = 1_000_000_000
SOURCES: list[Source] = [
    Source("raw_python", int(9.8 * B), fmt_code_file,
           hf_name="bigcode/the-stack-dedup", hf_data_dir="data/python", keep=keep_code_file,
           note="gated: accept terms on the Hub + `huggingface-cli login`. Field: content"),
    Source("python_edu", int(2 * B), fmt_code_file,
           hf_name="HuggingFaceTB/smollm-corpus", hf_config="python-edu", needs_local=True,
           note="the Hub copy lists blob_ids only; file contents must be fetched from the "
                "Software Heritage S3 bucket. Provide the result as jsonl ({'content': ...}) via --local"),
    Source("commits", int(1.5 * B), fmt_commit,
           hf_name="bigcode/commitpack", hf_data_dir="python", hf_revision="refs/convert/parquet", shuffle_buffer=2_000,
           note="loaded the way your 5B notebook did (data_dir + parquet revision). May be smaller than its budget; the notebook mix step caps it to what exists (up to 3 epochs)"),
    Source("stackoverflow", int(2 * B), fmt_text, needs_local=True,
           note="PYTHON-tagged Stack Overflow Q&A (Python 2 excluded) plus language-neutral SQL/regex/git/bash/algorithm questions, built by scripts/fetch_stackoverflow.py from the public "
                "posts parquet files (data up to Apr 2024, CC BY-SA). Provide via --local stackoverflow=<jsonl>. "
                "Check the current dump terms before training on it."),
    Source("fineweb_edu", int(2.5 * B), fmt_text,
           hf_name="HuggingFaceFW/fineweb-edu", hf_config="sample-10BT"),
    Source("wikipedia", int(0.5 * B), fmt_text,
           hf_name="wikimedia/wikipedia", hf_config="20231101.en"),
    Source("math", int(1 * B), fmt_text,
           hf_name="open-web-math/open-web-math"),
    Source("code_contests", int(0.15 * B), fmt_code_contests,
           hf_name="deepmind/code_contests", columns=["description", "solutions"], batch_size=8, shuffle_buffer=8,
           note="competitive programming (CC BY 4.0): statement + correct Python 3 solutions. SMALL: the "
                "real size is printed after tokenizing; the mix step uses at most 3 epochs of it"),
    Source("apps", int(0.05 * B), fmt_apps,
           hf_name="codeparrot/apps", hf_config=None, hf_revision="refs/convert/parquet",
           columns=["question", "solutions", "starter_code"], batch_size=64, shuffle_buffer=200,
           note="APPS problems + solutions (competitive/interview style). Loaded from the parquet "
                "conversion because the original is a loading script; verify with `probe`. SMALL like above"),
    Source("extras", int(0.5 * B), fmt_issue,
           hf_name="bigcode/the-stack-github-issues",
           note="OPTIONAL, OFF in the notebook by default: GitHub issues from ALL languages (there is no "
                "Python filter), so it would add Java/JS/C++ text to a Python model. Schema also uncertain "
                "-> check with `probe`."),
]
BY_NAME = {s.name: s for s in SOURCES}


# ----------------------------------------------------------------------------
def iter_rows(src: Source, local: dict[str, str], seed: int = 42, shuffle: bool = True) -> Iterator[dict]:
    if src.name in local:
        with open(local[src.name], "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
        return
    if src.needs_local:
        raise SystemExit(
            f"Source '{src.name}' has no text on the Hub: {src.note}\n"
            f"Pass --local {src.name}=/path/to/file.jsonl or exclude it with --skip {src.name}."
        )
    from datasets import load_dataset  # lazy import

    base = dict(split=src.split, streaming=True)
    if src.hf_revision:
        base["revision"] = src.hf_revision
    # Try the memory-saving options first; datasets that do not accept them (not parquet-backed)
    # raise TypeError, and we fall back to a plain load.
    attempts = []
    push = {k: v for k, v in (("columns", src.columns), ("batch_size", src.batch_size)) if v}
    if push:
        attempts.append(push)
        if "batch_size" in push:
            attempts.append({k: v for k, v in push.items() if k != "batch_size"})
    attempts.append({})
    ds, last_err = None, None
    config = src.hf_config
    for extra in attempts:
        for _try in range(2):
            try:
                ds = load_dataset(src.hf_name, config, data_dir=src.hf_data_dir, **base, **extra)
                break
            except TypeError as e:           # dataset does not accept the memory options: try fewer
                last_err = e
                break
            except ValueError as e:          # wrong config name: the error lists the real ones
                m = re.search(r"Available: \[(.*?)\]", str(e))
                names = re.findall(r"'([^']+)'", m.group(1)) if m else []
                if _try == 0 and names and config not in names:
                    print(f"  [{src.name}] config {config!r} not found; using {names[0]!r}", file=sys.stderr)
                    config = names[0]
                    continue
                raise
        if ds is not None:
            break
    if ds is None:
        raise last_err
    if src.columns and getattr(ds, "column_names", None) and not push_applied(ds, src.columns):
        try:
            ds = ds.select_columns(src.columns)
        except Exception:  # noqa: BLE001 - older datasets versions: just keep all columns
            pass
    if shuffle and src.shuffle_buffer > 0:
        # Buffered shuffle breaks up repo/date ordering inside each stream.
        ds = ds.shuffle(seed=seed, buffer_size=src.shuffle_buffer)
    yield from ds


def push_applied(ds, columns) -> bool:
    names = getattr(ds, "column_names", None)
    return bool(names) and set(names) == set(columns)


def iter_docs(src: Source, local: dict[str, str], skip_rows: int = 0) -> Iterator[tuple[int, str]]:
    """Yield (rows_consumed_so_far, formatted_text). skip_rows resumes a stream:
    the shuffled stream order is deterministic (fixed seed), so skipping the first
    N rows lands exactly where an interrupted run stopped."""
    rows = iter_rows(src, local)
    if skip_rows:
        # Skipping must still stream (download + decode) every skipped row, so it takes a while and
        # would otherwise look frozen: report progress.
        print(f"  [{src.name}] skipping the first {skip_rows:,} rows already done (no tokenizing, but the "
              f"data must be streamed again - this can take a while)...", file=sys.stderr, flush=True)
        t0 = time.time()
        for i, _ in enumerate(itertools.islice(rows, skip_rows), 1):
            if i % 500_000 == 0:
                rate = i / (time.time() - t0)
                print(f"  [{src.name}] skipped {i:,}/{skip_rows:,} rows ({rate:,.0f} rows/s, "
                      f"~{(skip_rows - i) / rate / 60:.0f} min left)", file=sys.stderr, flush=True)
        print(f"  [{src.name}] skip done in {(time.time() - t0) / 60:.1f} min; resuming", file=sys.stderr, flush=True)
    seen = skip_rows
    for row in rows:
        seen += 1
        try:
            if src.keep is not None and not src.keep(row):
                continue
            text = src.fmt(row)
        except KeyError:
            continue  # rows missing a field are skipped
        if text and text.strip():
            yield seen, text


def selected(args) -> list[Source]:
    only = set(getattr(args, "only", None) or [])
    if only - set(BY_NAME):
        raise SystemExit(f"Unknown --only name(s): {sorted(only - set(BY_NAME))}")
    skip = set(args.skip or [])
    if only:
        skip |= set(BY_NAME) - only
    unknown = skip - set(BY_NAME)
    if unknown:
        raise SystemExit(f"Unknown --skip name(s): {sorted(unknown)}; valid: {sorted(BY_NAME)}")
    return [s for s in SOURCES if s.name not in skip]


def parse_local(pairs: list[str] | None) -> dict[str, str]:
    out = {}
    for p in pairs or []:
        name, path = p.split("=", 1)
        if name not in BY_NAME:
            raise SystemExit(f"--local: unknown source '{name}'")
        out[name] = path
    return out


# ----------------------------------------------------------------------------
def cmd_plan(args) -> None:
    srcs = selected(args)
    total = sum(s.tokens for s in srcs)
    print(f"{'source':<14}{'tokens':>10}{'share':>8}   dataset")
    for s in srcs:
        where = (s.hf_name or "local jsonl (--local)") + (f" [{s.hf_config}]" if s.hf_config else "") + (f" ({s.hf_data_dir})" if s.hf_data_dir else "")
        print(f"{s.name:<14}{s.tokens/B:>9.2f}B{s.tokens/total:>8.1%}   {where}")
        if s.note:
            print(f"{'':<32}-> {s.note}")
    print(f"{'TOTAL':<14}{total/B:>9.2f}B")
    cmd = ["python scripts/build_data_mix.py --dtype uint16 --context-len 2048 --seed 42 \\",
           "    --out artifacts/mix_20b/corpus.bin \\"]
    for s in srcs:
        cmd.append(f"    --source {args.out}/{s.name}.bin:{s.tokens/total:.4f} \\")
    cmd[-1] = cmd[-1].rstrip(" \\")
    print("\nThen blend with:\n" + "\n".join(cmd))
    print("\n(Ratios are re-normalised over the sources you did not --skip; they must sum to 1.)")


def cmd_probe(args) -> None:
    local = parse_local(args.local)
    for s in selected(args):
        print("=" * 78)
        print(f"{s.name}: {s.hf_name} {s.hf_config or ''} {s.hf_data_dir or ''}")
        try:
            fields, looked, kept, sample = None, 0, 0, None
            # look at the first rows (no shuffle buffer: cheap on RAM); some rows are legitimately
            # filtered out (e.g. a problem with no Python 3 solution), so do not stop at row 1
            for row in itertools.islice(iter_rows(s, local, shuffle=False), args.rows):
                looked += 1
                fields = fields or sorted(row)
                try:
                    if s.keep is not None and not s.keep(row):
                        continue
                    text = s.fmt(row)
                except KeyError:
                    continue
                if text and text.strip():
                    kept += 1
                    sample = sample or text
            print("fields:", fields)
            print(f"rows inspected: {looked} | passed the filters + formatting: {kept}")
            if sample:
                print("formatted (first 300 chars):", repr(sample[:300]))
            else:
                print("NO row produced any text - check the source's fmt/keep functions against the fields above")
        except SystemExit as e:
            print("NEEDS ACTION:", e)
        except Exception as e:  # noqa: BLE001 - we want to show anything that goes wrong
            print(f"FAILED: {type(e).__name__}: {e}")


def cmd_sample(args) -> None:
    local = parse_local(args.local)
    srcs = selected(args)
    total = sum(s.tokens for s in srcs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget_bytes = args.mb * 1_000_000
    failed = []
    for s in srcs:
        want = int(budget_bytes * s.tokens / total)
        path = out / f"{s.name}.txt"
        written = 0
        try:
            with open(path, "w", encoding="utf-8") as f:
                for _, text in iter_docs(s, local):
                    f.write(text + "\n")
                    written += len(text.encode("utf-8"))
                    if written >= want:
                        break
            print(f"{s.name}: {written/1e6:.0f} MB -> {path}")
        except (Exception, SystemExit) as e:  # noqa: BLE001 - report it, keep sampling the others
            path.unlink(missing_ok=True)
            failed.append(s)
            print(f"WARNING: could not sample '{s.name}': {type(e).__name__}: {e}", file=sys.stderr)
    if failed:
        important = [s.name for s in failed if s.tokens / TOTAL_TOKENS >= 0.02]
        if important:
            raise SystemExit(f"Sampling failed for important source(s) {important}; fix them before training the tokenizer.")
        print("Continuing: the failed source(s) are each <2% of the mix, so the tokenizer is not noticeably "
              "worse without them: " + ", ".join(f"{s.name} ({s.tokens / TOTAL_TOKENS:.2%})" for s in failed))


def _atomic_copy(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst_dir / (src.name + ".part")
    shutil.copyfile(src, tmp)
    if tmp.stat().st_size != src.stat().st_size:
        raise IOError(f"copy of {src} to {dst_dir} is incomplete")
    os.replace(tmp, dst_dir / src.name)


def _write_progress(path: Path, prog: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(prog))
    os.replace(tmp, path)


def cmd_tokenize(args) -> None:
    """Tokenize each source into 1B-token shards: <out>/<name>/shard_0000.bin ...

    After every finished shard, progress.json is updated and (with --sync-dir,
    e.g. a Google Drive folder) the shard + progress are copied there. If the
    runtime dies, re-running the same command resumes from the last finished
    shard (at most one shard of work is lost). Run `merge` afterwards to get the
    single <name>.bin that build_data_mix.py expects.
    """
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    if tok.get_vocab_size() > 65535:
        raise SystemExit("vocab_size > 65535: uint16 .bin files cannot hold it.")
    for needed in ("<file_sep>", "<commit_before>", "<commit_message>", "<question>"):
        if tok.token_to_id(needed) is None:
            raise SystemExit(
                f"Tokenizer has no '{needed}' special token. Retrain it with the updated "
                "SPECIAL_TOKENS in data/tokenizer.py (the old tokenizer would fragment these tags)."
            )
    local = parse_local(args.local)
    out = Path(args.out)
    sync = Path(args.sync_dir) if args.sync_dir else None

    for s in selected(args):
        sdir = out / s.name
        sdir.mkdir(parents=True, exist_ok=True)
        prog_path = sdir / "progress.json"
        if not prog_path.exists() and sync and (sync / s.name / "progress.json").exists():
            shutil.copyfile(sync / s.name / "progress.json", prog_path)   # resume after a wiped runtime
        prog = json.loads(prog_path.read_text()) if prog_path.exists() else \
            {"tokens": 0, "rows": 0, "shards": 0, "done": False}
        budget = int(s.tokens * args.scale)
        if prog["done"] or prog["tokens"] >= budget:
            print(f"[skip] {s.name}: {prog['tokens']:,} tokens already done")
            continue
        if prog["tokens"]:
            print(f"[resume] {s.name}: {prog['tokens']:,} tokens / {prog['shards']} shards done, "
                  f"skipping first {prog['rows']:,} rows")

        t0, base = time.time(), prog["tokens"]
        batch: list[str] = []
        shard_tokens, shard_f, shard_path = 0, None, None
        rows_seen = prog["rows"]

        def open_shard():
            nonlocal shard_f, shard_path, shard_tokens
            shard_path = sdir / f"shard_{prog['shards']:04d}.bin"
            shard_f = open(str(shard_path) + ".part", "wb")
            shard_tokens = 0

        def close_shard(done: bool) -> None:
            nonlocal shard_f
            shard_f.close()
            os.replace(str(shard_path) + ".part", shard_path)
            prog["tokens"] += shard_tokens
            prog["rows"] = rows_seen
            prog["shards"] += 1
            prog["done"] = done
            _write_progress(prog_path, prog)
            if sync:
                _atomic_copy(shard_path, sync / s.name)
                _atomic_copy(prog_path, sync / s.name)
            shard_f = None
            rate = (prog["tokens"] - base) / max(time.time() - t0, 1e-9)
            print(f"  {s.name}: shard {prog['shards']-1} done | {prog['tokens']/1e6:,.0f}M / "
                  f"{budget/1e6:,.0f}M tokens | {rate/1e3:,.0f}k tok/s", flush=True)

        def flush() -> None:
            nonlocal batch, shard_tokens
            enc = tok.encode_batch(batch)          # multi-threaded in Rust
            ids = np.fromiter((i for e in enc for i in e.ids), dtype=np.uint16)
            shard_f.write(ids.tobytes())
            shard_tokens += len(ids)
            batch = []

        open_shard()
        for rows_seen, text in iter_docs(s, local, skip_rows=prog["rows"]):
            batch.append(text)
            if len(batch) >= args.batch_docs:
                flush()
                if prog["tokens"] + shard_tokens >= budget:
                    break
                if shard_tokens >= args.shard_tokens:
                    close_shard(done=False)
                    open_shard()
        if batch:
            flush()
        close_shard(done=True)
        print(f"[done] {s.name}: {prog['tokens']:,} tokens in {prog['shards']} shards")
        if prog["tokens"] < budget * 0.95:
            print(f"  WARNING: only {prog['tokens']/budget:.0%} of the {budget:,}-token budget was "
                  f"available. The notebook's mix step uses what exists (repeating small sources only up to a cap).")


def cmd_merge(args) -> None:
    """Concatenate <out>/<name>/shard_*.bin into <out>/<name>.bin. Shards missing
    locally are fetched one at a time from --sync-dir and deleted again afterwards,
    so this never needs 2x the disk."""
    out = Path(args.out)
    sync = Path(args.sync_dir) if args.sync_dir else None
    for s in selected(args):
        sdir = out / s.name
        prog_path = sdir / "progress.json"
        if not prog_path.exists() and sync and (sync / s.name / "progress.json").exists():
            sdir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sync / s.name / "progress.json", prog_path)
        if not prog_path.exists():
            raise SystemExit(f"{s.name}: no progress.json - run `tokenize` first.")
        prog = json.loads(prog_path.read_text())
        if not prog["done"]:
            if not args.allow_partial:
                raise SystemExit(f"{s.name}: tokenization not finished ({prog['tokens']:,} tokens so far). "
                                 f"To use only the finished shards, pass --allow-partial.")
            print(f"[partial] {s.name}: using the {prog['shards']} finished shard(s), {prog['tokens']:,} tokens")
        final = out / f"{s.name}.bin"
        total = 0
        with open(str(final) + ".part", "wb") as fo:
            for i in range(prog["shards"]):
                name = f"shard_{i:04d}.bin"
                shard = sdir / name
                fetched = False
                if not shard.exists():
                    if not (sync and (sync / s.name / name).exists()):
                        raise SystemExit(f"{s.name}: {name} is neither local nor in --sync-dir")
                    shutil.copyfile(sync / s.name / name, shard)
                    fetched = True
                with open(shard, "rb") as fi:
                    shutil.copyfileobj(fi, fo, length=64 * 1024 * 1024)
                total += shard.stat().st_size // 2
                if fetched:
                    shard.unlink()
        if total != prog["tokens"]:
            raise SystemExit(f"{s.name}: merged {total:,} tokens but progress says {prog['tokens']:,}")
        os.replace(str(final) + ".part", final)
        print(f"[merged] {s.name}: {total:,} tokens -> {final}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, out_default):
        p.add_argument("--skip", nargs="*", help="source names to leave out")
        p.add_argument("--only", nargs="*", help="process just these sources (e.g. one per Colab cell)")
        p.add_argument("--local", action="append", help="name=path.jsonl for sources not streamable from the Hub")
        p.add_argument("--out", default=out_default)

    p = sub.add_parser("plan"); common(p, "artifacts/src_20b"); p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("probe"); common(p, "")
    p.add_argument("--rows", type=int, default=200, help="how many leading rows to inspect per source")
    p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("sample"); common(p, "artifacts/tok_sample")
    p.add_argument("--mb", type=int, default=1000, help="total sample size in MB"); p.set_defaults(fn=cmd_sample)
    p = sub.add_parser("tokenize"); common(p, "artifacts/src_20b")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--batch-docs", type=int, default=1000)
    p.add_argument("--shard-tokens", type=int, default=1_000_000_000,
                   help="tokens per shard; each finished shard is a resume point")
    p.add_argument("--sync-dir", default=None,
                   help="e.g. a Drive folder: finished shards + progress are copied here, and "
                        "progress is restored from here after a wiped runtime")
    p.add_argument("--scale", type=float, default=1.0,
                   help="multiply every token budget (e.g. 0.01 for a 1%% dry run)")
    p.set_defaults(fn=cmd_tokenize)
    p = sub.add_parser("merge"); common(p, "artifacts/src_20b")
    p.add_argument("--sync-dir", default=None)
    p.add_argument("--allow-partial", action="store_true",
                   help="merge only the finished shards of a source whose tokenization was stopped early")
    p.set_defaults(fn=cmd_merge)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
