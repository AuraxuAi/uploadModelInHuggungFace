"""
Build REPOSITORY-LEVEL training documents from Python libraries on GitHub.

One output document = one repository:

    <repo_name>owner/name<file_sep>README.md
    ...readme...<file_sep>pkg/utils.py
    ...code...<file_sep>pkg/api.py
    ...code that imports pkg.utils...

Files inside a repo are ordered so that a file's in-repo imports come BEFORE it
(dependency order, tests last). That is what lets a model learn "when I see
`from pkg.utils import helper`, the definition of `helper` appeared earlier in
my context" - the point of repo-level training. For this to help, the model must
be trained with a context window long enough to span several files (see
configs/code_500m_stage2_repo.yaml: 8192 tokens).

Pipeline (three subcommands, each resumable):

  list    Resolve libraries to GitHub repos. Default: the most-downloaded PyPI
          packages (hugovk/top-pypi-packages, 15k rows) -> PyPI JSON API ->
          the repo URL in the package metadata. Or give your own --package-list
          (one PyPI name, "owner/repo", or GitHub URL per line).
  clone   Shallow-clone each repo, check its LICENSE, filter files, order them,
          and write jsonl parts (optionally copying each finished part to a
          --sync-dir such as Google Drive).
  merge   Concatenate the parts into one jsonl (+ a manifest.csv listing repo,
          commit and license for attribution/auditing).

Licensing - please read
  * Only repositories whose LICENSE file starts with a permissive license
    (MIT, BSD, Apache-2.0, ISC, PSF, Unlicense) are kept. Copyleft (GPL/LGPL/AGPL/MPL...),
    unknown or missing licenses are skipped. This is a keyword heuristic on the
    top of the main license file, not legal review: spot-check manifest.csv.
  * Permissive licenses still require keeping copyright/license notices when the
    code is redistributed. Whether and how that applies to model training is your
    call (not legal advice); the manifest records repo + commit + license per repo.
  * Nothing from the cloned repos is ever executed; files are only read and parsed.

Usage:
  python scripts/fetch_github_repos.py list  --top 5000 --out artifacts/repos.json
  python scripts/fetch_github_repos.py clone --repos artifacts/repos.json --out-dir artifacts/github_repos \\
        --sync-dir /content/drive/MyDrive/transformer_project_20b/github_repos
  python scripts/fetch_github_repos.py merge --out-dir artifacts/github_repos --dest artifacts/github_repos.jsonl \\
        --sync-dir /content/drive/MyDrive/transformer_project_20b/github_repos
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import heapq
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

warnings.simplefilter("ignore", SyntaxWarning)   # old code is full of invalid escape sequences

TOP_PYPI_URLS = [
    "https://raw.githubusercontent.com/hugovk/top-pypi-packages/main/top-pypi-packages.min.json",
    "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json",
]
GITHUB_RE = re.compile(r"github\.com[/:]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", re.I)

# ----------------------------------------------------------------------------
# License classification (keyword heuristics - see module docstring)
# ----------------------------------------------------------------------------
_COPYLEFT = re.compile(
    r"\b(a|l)?gpl\b|gnu (affero |lesser |library )?general public|mozilla public|\bmpl\b|\beupl\b|"
    r"european union public|cecill|proprietary|commons clause|non-?commercial|\bcc[- ]by-nc|"
    r"server side public|\bsspl\b|business source|elastic license", re.I)
_PERMISSIVE = re.compile(
    r"\bmit\b|apache|\bbsd\b|\bisc\b|python software foundation|\bpsf\b|unlicense|public domain|"
    r"\bcc0\b|\bzlib\b|boost software", re.I)


def classify_metadata_license(text: str) -> Optional[str]:
    """'copyleft' / 'permissive' / None from PyPI classifiers or an SPDX string."""
    if not text:
        return None
    if _COPYLEFT.search(text):
        return "copyleft"
    if _PERMISSIVE.search(text):
        return "permissive"
    return None


def detect_license_file(repo_dir: Path) -> tuple[str, bool]:
    """Classify the repo's main license file by its opening text. Returns (label, allowed)."""
    cands = sorted(p for p in repo_dir.iterdir()
                   if p.is_file() and re.match(r"(?i)(licen[cs]e|copying|unlicense)", p.name))
    if not cands:
        return "none", False
    head = cands[0].read_text(encoding="utf-8", errors="ignore")[:2500].lower()
    if re.search(r"gnu (affero |lesser |library )?general public|mozilla public|european union public|"
                 r"server side public|business source|commons clause|non-?commercial", head):
        return "copyleft", False
    if "apache license" in head and "2.0" in head:
        return "Apache-2.0", True
    if "permission is hereby granted, free of charge" in head:
        return "MIT", True
    if "redistribution and use in source and binary forms" in head:
        return "BSD", True
    if "permission to use, copy, modify, and/or distribute this software" in head:
        return "ISC", True
    if "python software foundation" in head or "psf license" in head:
        return "PSF", True
    if "unencumbered software released into the public domain" in head:
        return "Unlicense", True
    return "unknown", False


# ----------------------------------------------------------------------------
# `list`: libraries -> GitHub repos
# ----------------------------------------------------------------------------
def http_get(url: str, retries: int = 3, timeout: int = 30) -> Optional[bytes]:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "transformer-project/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (attempt + 1))
        except Exception:  # noqa: BLE001 - transient network trouble: retry
            time.sleep(2 * (attempt + 1))
    return None


def github_repo_from_urls(urls: list[str]) -> Optional[str]:
    for u in urls:
        m = GITHUB_RE.search(u or "")
        if m:
            owner, name = m.group(1), m.group(2)
            name = re.sub(r"\.git$", "", name)
            if name and owner.lower() not in {"sponsors", "orgs", "topics"}:
                return f"{owner}/{name}"
    return None


def resolve_package(name: str) -> dict:
    data = http_get(f"https://pypi.org/pypi/{name}/json")
    rec = {"package": name, "repo": None, "pre_license": None}
    if not data:
        return rec
    info = json.loads(data).get("info", {})
    urls = info.get("project_urls") or {}
    preferred = [v for k, v in urls.items()
                 if re.search(r"(?i)source|repo|code|github", k)] + list(urls.values()) + [info.get("home_page") or ""]
    rec["repo"] = github_repo_from_urls(preferred)
    lic_cls = " ".join(c for c in (info.get("classifiers") or []) if c.startswith("License"))
    rec["pre_license"] = classify_metadata_license(" ".join([info.get("license_expression") or "", lic_cls,
                                                            (info.get("license") or "")[:200]]))
    return rec


def cmd_list(args) -> None:
    items: list[dict] = []   # {"package":..., "downloads":...} or {"repo":...}
    if args.package_list:
        for line in Path(args.package_list).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            direct = github_repo_from_urls([line]) if ("github.com" in line) else (line if re.fullmatch(r"[\w.-]+/[\w.-]+", line) else None)
            items.append({"repo": direct, "pre_license": None, "package": None} if direct else {"package": line})
    else:
        raw = None
        for u in TOP_PYPI_URLS:
            raw = http_get(u)
            if raw:
                break
        if not raw:
            raise SystemExit("Could not download the top-PyPI-packages list; pass --package-list instead.")
        rows = json.loads(raw)["rows"][: args.top]
        items = [{"package": r["project"], "downloads": r["download_count"]} for r in rows]

    todo = [it for it in items if it.get("package") and not it.get("repo")]
    print(f"resolving {len(todo)} packages via PyPI ({args.workers} threads)...", flush=True)
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(resolve_package, it["package"]): it for it in todo}
        for i, f in enumerate(as_completed(futs), 1):
            it = futs[f]
            results[it["package"]] = f.result()
            if i % 250 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    repos: dict[str, dict] = {}
    unresolved = 0
    for it in items:
        rec = it if it.get("repo") else results.get(it["package"], {})
        repo = rec.get("repo")
        if not repo:
            unresolved += 1
            continue
        key = repo.lower()
        entry = repos.setdefault(key, {"repo": repo, "packages": [], "downloads": 0, "pre_license": rec.get("pre_license")})
        if it.get("package"):
            entry["packages"].append(it["package"])
        entry["downloads"] += it.get("downloads", 0)
        if rec.get("pre_license") == "copyleft":
            entry["pre_license"] = "copyleft"
    out = sorted(repos.values(), key=lambda e: -e["downloads"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    n_cl = sum(1 for e in out if e["pre_license"] == "copyleft")
    print(f"{len(out)} unique GitHub repos ({unresolved} packages had no GitHub URL); "
          f"{n_cl} flagged copyleft by PyPI metadata (will be skipped) -> {args.out}")


# ----------------------------------------------------------------------------
# `clone`: per-repo processing (runs in worker processes)
# ----------------------------------------------------------------------------
SKIP_DIRS_ANYWHERE = {".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env", "site-packages", "__pycache__",
                      ".tox", ".eggs", "_vendor", "vendor", "vendored", "third_party", "thirdparty", "extern",
                      "external", "migrations", ".mypy_cache", ".pytest_cache"}
SKIP_DIRS_TOPLEVEL = {"build", "dist"}
GENERATED = ("auto-generated", "autogenerated", "do not edit", "generated by the protocol buffer",
             "this file was generated", "@generated", "generated by grpc")
TEST_DIRS = {"test", "tests", "testing"}


def is_test_path(rel: str) -> bool:
    parts = rel.split("/")
    name = parts[-1]
    return (any(p in TEST_DIRS for p in parts[:-1]) or name.startswith("test_") or name.endswith("_test.py")
            or name == "conftest.py")


def keep_source(rel: str, text: str) -> bool:
    n = len(text)
    if n < 50 or n > 400_000:
        return False
    if rel.endswith(("_pb2.py", "_pb2_grpc.py")):
        return False
    head = text[:500].lower()
    if any(g in head for g in GENERATED):
        return False
    lines = text.splitlines()
    if not lines or max(map(len, lines)) > 1000 or n / len(lines) > 120:
        return False
    if sum(c.isalnum() for c in text) / n < 0.25:
        return False
    try:
        ast.parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False   # python 2 files, templates, broken files
    return True


def walk_python(repo_dir: Path, max_files: int):
    """Return ({rel: text} for kept .py files, set of package dirs that contain __init__.py)."""
    files: dict[str, str] = {}
    pkg_dirs: set[str] = set()
    seen_hashes: set[str] = set()
    for root, dirs, names in os.walk(repo_dir, followlinks=False):
        rel_root = os.path.relpath(root, repo_dir).replace(os.sep, "/")
        rel_root = "" if rel_root == "." else rel_root
        dirs[:] = sorted(d for d in dirs
                         if d not in SKIP_DIRS_ANYWHERE and not (rel_root == "" and d in SKIP_DIRS_TOPLEVEL)
                         and not os.path.islink(os.path.join(root, d)))
        if "__init__.py" in names:
            pkg_dirs.add(rel_root)
        for name in sorted(names):
            if not name.endswith(".py") or len(files) >= max_files:
                continue
            full = os.path.join(root, name)
            if os.path.islink(full):
                continue
            rel = f"{rel_root}/{name}" if rel_root else name
            try:
                text = Path(full).read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if not keep_source(rel, text):
                continue
            h = hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            files[rel] = text
    return files, pkg_dirs


def dotted_name(rel: str, pkg_dirs: set[str]) -> str:
    """Import name of a file, given which directories are packages (have __init__.py)."""
    d, _, fname = rel.rpartition("/")
    stem = fname[:-3]
    parts: list[str] = [] if stem == "__init__" else [stem]
    while d in pkg_dirs:
        parts.insert(0, d.rpartition("/")[2])
        d = d.rpartition("/")[0]
        if d == "" and "" not in pkg_dirs:
            break
    return ".".join(parts)


def build_dependencies(files: dict[str, str], pkg_dirs: set[str]) -> dict[str, set[str]]:
    index: dict[str, str] = {}
    for rel in sorted(files, key=lambda r: (is_test_path(r), r)):
        index.setdefault(dotted_name(rel, pkg_dirs), rel)

    deps: dict[str, set[str]] = {r: set() for r in files}
    for rel, text in files.items():
        me = dotted_name(rel, pkg_dirs)
        is_pkg_init = rel.endswith("/__init__.py") or rel == "__init__.py"
        pkg = me if is_pkg_init else me.rpartition(".")[0]
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                for a in node.names:
                    targets.append(a.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base_parts = pkg.split(".") if pkg else []
                    if node.level - 1 > len(base_parts):
                        continue
                    base_parts = base_parts[: len(base_parts) - (node.level - 1)]
                    base = ".".join(base_parts + ([node.module] if node.module else []))
                else:
                    base = node.module or ""
                for a in node.names:
                    sub = f"{base}.{a.name}" if base else a.name
                    targets.append(sub if sub in index else base)
            for t in targets:
                # longest existing prefix = the most specific module that was imported
                parts = t.split(".")
                for i in range(len(parts), 0, -1):
                    cand = ".".join(parts[:i])
                    if cand in index:
                        dep = index[cand]
                        if dep != rel:
                            deps[rel].add(dep)
                        break
    return deps


AUX_DIRS = {"docs", "doc", "examples", "example", "benchmarks", "benchmark", "scripts", "tools", "tutorials", "demo", "demos"}


def order_rank(rel: str) -> int:
    """0 = library code, 1 = docs/examples/setup scripts, 2 = tests."""
    if is_test_path(rel):
        return 2
    parts = rel.split("/")
    if parts[0] in AUX_DIRS or parts[-1] in {"setup.py", "conf.py", "noxfile.py", "tasks.py", "fabfile.py"}:
        return 1
    return 0


def _toposort(nodes: set[str], deps: dict[str, set[str]]) -> list[str]:
    """Dependencies first (edges to files outside `nodes` are ignored); cycles are broken by
    releasing the file with the fewest unmet dependencies; ties resolved by path."""
    rev: dict[str, list[str]] = {r: [] for r in nodes}
    indeg = {r: 0 for r in nodes}
    for r in nodes:
        for d in deps[r]:
            if d in nodes:
                indeg[r] += 1
                rev[d].append(r)
    heap = [r for r, n in indeg.items() if n == 0]
    heapq.heapify(heap)
    emitted: list[str] = []
    done: set[str] = set()
    remaining = set(nodes)
    while remaining:
        if not heap:   # import cycle
            heapq.heappush(heap, min(remaining, key=lambda x: (indeg[x], x)))
        r = heapq.heappop(heap)
        if r in done:
            continue
        done.add(r)
        remaining.discard(r)
        emitted.append(r)
        for nxt in rev[r]:
            if nxt not in done:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    heapq.heappush(heap, nxt)
    return emitted


def dependency_order(files: dict[str, str], pkg_dirs: set[str]) -> list[str]:
    """Library code first (imports before importers), then docs/examples, then tests."""
    deps = build_dependencies(files, pkg_dirs)
    out: list[str] = []
    for rank in (0, 1, 2):
        out += _toposort({r for r in files if order_rank(r) == rank}, deps)
    return out


def read_readme(repo_dir: Path, limit: int = 3000) -> Optional[tuple[str, str]]:
    for p in sorted(repo_dir.iterdir()):
        if p.is_file() and re.fullmatch(r"(?i)readme(\.(md|rst|txt))?", p.name):
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
            if len(text) >= 50:
                return p.name, text[:limit]
    return None


def clone_repo(repo: str, dest: Path, timeout: int) -> Optional[str]:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
    for _ in range(2):
        try:
            r = subprocess.run(["git", "clone", "--depth", "1", "--single-branch", "--no-tags", "-q",
                                "-c", "core.symlinks=false", f"https://github.com/{repo}.git", str(dest)],
                               capture_output=True, timeout=timeout, env=env)
            if r.returncode == 0:
                return subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True,
                                      text=True).stdout.strip()
        except subprocess.TimeoutExpired:
            pass
        shutil.rmtree(dest, ignore_errors=True)
        time.sleep(3)
    return None


def process_repo(job: dict) -> dict:
    repo = job["repo"]
    res = {"repo": repo, "status": "", "license": "", "commit": "", "n_files": 0, "chars": 0, "text": None}
    tmp = Path(tempfile.mkdtemp(prefix="gh_", dir=job["workdir"]))
    try:
        dest = tmp / "r"
        commit = clone_repo(repo, dest, job["clone_timeout"])
        if not commit:
            res["status"] = "skip:clone"
            return res
        res["commit"] = commit
        label, allowed = detect_license_file(dest)
        res["license"] = label
        if not allowed:
            res["status"] = f"skip:license:{label}"
            return res
        files, pkg_dirs = walk_python(dest, job["max_files"])
        if len(files) < job["min_files"]:
            res["status"] = "skip:few_files"
            return res
        order = dependency_order(files, pkg_dirs)
        parts = [f"<repo_name>{repo}"]
        total = len(parts[0])
        readme = read_readme(dest)
        if readme:
            parts.append(f"<file_sep>{readme[0]}\n{readme[1]}")
            total += len(parts[-1])
        n = 0
        for rel in order:
            piece = f"<file_sep>{rel}\n{files[rel]}"
            if total + len(piece) > job["max_chars"]:
                break   # tests sit at the end of the order, so they are dropped first
            parts.append(piece)
            total += len(piece)
            n += 1
        res.update(status="ok", n_files=n, chars=total, text="".join(parts))
        return res
    except Exception as e:  # noqa: BLE001 - one bad repo must not kill the whole run
        res["status"] = f"skip:error:{type(e).__name__}"
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _atomic_copy(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst_dir / (src.name + ".part")
    shutil.copyfile(src, tmp)
    if tmp.stat().st_size != src.stat().st_size:
        raise IOError(f"incomplete copy of {src}")
    os.replace(tmp, dst_dir / src.name)


def _save_progress(path: Path, prog: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(prog))
    os.replace(tmp, path)


def cmd_clone(args) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sync = Path(args.sync_dir) if args.sync_dir else None
    prog_path = out / "progress.json"
    if not prog_path.exists() and sync and (sync / "progress.json").exists():
        shutil.copyfile(sync / "progress.json", prog_path)      # resume after a wiped runtime
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {"parts": 0, "repos": {}}

    repos = json.loads(Path(args.repos).read_text())
    if args.max_repos:
        repos = repos[: args.max_repos]
    jobs = []
    for e in repos:
        st = prog["repos"].get(e["repo"], {}).get("status")
        if st and st != "skip:clone":          # finished earlier (failed clones are retried)
            continue
        if e.get("pre_license") == "copyleft" and not args.ignore_pypi_license:
            prog["repos"][e["repo"]] = {"status": "skip:license_pypi"}
            continue
        jobs.append({"repo": e["repo"], "workdir": str(out), "max_chars": args.max_repo_chars,
                     "min_files": args.min_files, "max_files": args.max_files,
                     "clone_timeout": args.clone_timeout})
    print(f"{len(jobs)} repos to process ({len(prog['repos'])} already recorded, {prog['parts']} parts done)", flush=True)

    part_idx = prog["parts"]
    part_f = None
    part_path = None
    part_bytes = 0
    in_part: dict[str, dict] = {}
    stats = {"ok": 0, "skipped": 0}
    t0 = time.time()

    def open_part():
        nonlocal part_f, part_path, part_bytes
        part_path = out / f"part_{part_idx:04d}.jsonl"
        part_f = open(part_path, "w", encoding="utf-8")
        part_bytes = 0

    def close_part():
        nonlocal part_f, part_idx
        part_f.close()
        prog["parts"] = part_idx + 1
        prog["repos"].update(in_part)
        in_part.clear()
        _save_progress(prog_path, prog)
        if sync:
            _atomic_copy(part_path, sync)
            _atomic_copy(prog_path, sync)
        part_idx += 1
        part_f = None
        print(f"  part {part_idx - 1} done | ok {stats['ok']} skipped {stats['skipped']} | "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    open_part()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process_repo, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            meta = {k: r[k] for k in ("status", "license", "commit", "n_files", "chars")}
            if r["status"] == "ok":
                line = json.dumps({"text": r["text"], "repo": r["repo"], "license": r["license"],
                                   "commit": r["commit"], "n_files": r["n_files"]}, ensure_ascii=False) + "\n"
                part_f.write(line)
                part_bytes += len(line.encode("utf-8"))
                stats["ok"] += 1
            else:
                stats["skipped"] += 1
            in_part[r["repo"]] = meta
            if part_bytes >= args.part_mb * 1_000_000:
                close_part()
                open_part()
    if in_part or part_bytes:
        close_part()
    else:
        part_f.close()
        part_path.unlink(missing_ok=True)
        if in_part == {} and jobs == []:
            _save_progress(prog_path, prog)
    ok_chars = sum(m.get("chars", 0) for m in prog["repos"].values() if m.get("status") == "ok")
    print(f"done: {sum(1 for m in prog['repos'].values() if m.get('status') == 'ok')} repos kept, "
          f"~{ok_chars / 3.5 / 1e9:.2f}B tokens (3.5 chars/token estimate)")
    from collections import Counter
    print("skip reasons:", dict(Counter(m["status"] for m in prog["repos"].values() if m.get("status") != "ok")))


def cmd_merge(args) -> None:
    out = Path(args.out_dir)
    sync = Path(args.sync_dir) if args.sync_dir else None
    prog_path = out / "progress.json"
    if not prog_path.exists() and sync and (sync / "progress.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sync / "progress.json", prog_path)
    prog = json.loads(prog_path.read_text())
    Path(args.dest).parent.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    with open(args.dest, "w", encoding="utf-8") as fo:
        for i in range(prog["parts"]):
            name = f"part_{i:04d}.jsonl"
            p = out / name
            fetched = False
            if not p.exists():
                if not (sync and (sync / name).exists()):
                    raise SystemExit(f"{name} is neither local nor in --sync-dir")
                shutil.copyfile(sync / name, p)
                fetched = True
            with open(p, "r", encoding="utf-8") as fi:
                for line in fi:
                    fo.write(line)
                    n_docs += 1
            if fetched:
                p.unlink()
    man = Path(args.dest).with_suffix(".manifest.csv")
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["repo", "commit", "license", "n_files", "chars"])
        for repo, m in sorted(prog["repos"].items()):
            if m.get("status") == "ok":
                w.writerow([repo, m["commit"], m["license"], m["n_files"], m["chars"]])
    print(f"merged {n_docs} repo documents -> {args.dest}\nmanifest -> {man}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list")
    p.add_argument("--top", type=int, default=5000, help="how many of the most-downloaded PyPI packages")
    p.add_argument("--package-list", help="file with one PyPI name / owner/repo / GitHub URL per line")
    p.add_argument("--out", default="artifacts/repos.json")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("clone")
    p.add_argument("--repos", default="artifacts/repos.json")
    p.add_argument("--out-dir", default="artifacts/github_repos")
    p.add_argument("--sync-dir", default=None, help="e.g. Drive folder: finished parts + progress are copied here")
    p.add_argument("--workers", type=int, default=max(2, min(8, (os.cpu_count() or 2) * 2)))
    p.add_argument("--max-repos", type=int, default=0)
    p.add_argument("--part-mb", type=int, default=300, help="size of each output part / resume point")
    p.add_argument("--max-repo-chars", type=int, default=4_000_000, help="cap per repo document (~1M tokens)")
    p.add_argument("--min-files", type=int, default=3)
    p.add_argument("--max-files", type=int, default=20_000)
    p.add_argument("--clone-timeout", type=int, default=300)
    p.add_argument("--ignore-pypi-license", action="store_true",
                   help="do not pre-skip repos whose PyPI metadata looks copyleft (the LICENSE-file check still applies)")
    p.set_defaults(fn=cmd_clone)

    p = sub.add_parser("merge")
    p.add_argument("--out-dir", default="artifacts/github_repos")
    p.add_argument("--dest", default="artifacts/github_repos.jsonl")
    p.add_argument("--sync-dir", default=None)
    p.set_defaults(fn=cmd_merge)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
