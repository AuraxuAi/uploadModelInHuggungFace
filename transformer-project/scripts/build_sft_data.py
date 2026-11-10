"""
Build the supervised fine-tuning file (jsonl of {"prompt", "response", "kind"}) from data you
already use, so no new licenses are involved. The prompt/response formats are EXACTLY the
formats the model saw in pretraining, only the loss is restricted to the response (and an end
token is added so it learns to stop):

  function   <file_sep>{imports}\\n\\n\\n{signature + docstring}      ->  {function body}
  stackoverflow <question>{title + question}<answer>                ->  {answer}
  commit     <commit_before>{old file}<commit_message>{msg}<commit_after>  ->  {new file}
  contest    <question>{problem statement}<answer>                  ->  {a correct Python 3 solution}

Boundary detail that matters: a prompt must end where the tokenizer would also split. Function
prompts therefore come in two variants - ending right after the docstring/signature (the response
then starts with the newline + indentation, exactly as in pretraining) and ending with a newline
(response starts at the indentation) - so the model handles both ways you may end a prompt.

Sources (streamed from Hugging Face, same loaders as the corpus build; needs HF_TOKEN):
  raw_python   -> function pairs        commits -> commit pairs
  code_contests, apps -> contest pairs  --so-jsonl (your stackoverflow.jsonl) -> Q&A pairs
For offline tests, any of them can be a local jsonl: --local raw_python=file.jsonl

Usage:
  python scripts/build_sft_data.py --so-jsonl artifacts/stackoverflow.jsonl --out artifacts/sft/sft_pairs.jsonl
"""
from __future__ import annotations

import argparse
import ast
import builtins
import difflib
import hashlib
import json
import random
import symtable
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_corpus_20b as bc  # noqa: E402  (source definitions + HF streaming)


# ----------------------------------------------------------------------------
# function pairs (signature + docstring -> body), from real Python files
# ----------------------------------------------------------------------------
def _is_placeholder_body(stmts) -> bool:
    """True when the body (after the docstring) is empty or only stubs: pass, ..., raise NotImplementedError."""
    if not stmts:
        return True
    for st in stmts:
        if isinstance(st, ast.Pass):
            continue
        if isinstance(st, ast.Expr) and isinstance(getattr(st, "value", None), ast.Constant) and st.value.value is Ellipsis:
            continue
        if isinstance(st, ast.Raise) and st.exc is not None:
            target = st.exc.func if isinstance(st.exc, ast.Call) else st.exc
            if isinstance(target, ast.Name) and target.id == "NotImplementedError":
                continue
        return False
    return True


def _free_global_names(fn_source: str) -> set[str] | None:
    """Names a function reads from the MODULE scope (not params, locals, comprehension vars, nested scopes).
    Uses the compiler's own symbol table, so comprehensions/lambdas/nested defs are handled correctly."""
    try:
        top = symtable.symtable(fn_source, "<fn>", "exec")
        fn_table = top.get_children()[0]
    except (SyntaxError, ValueError, IndexError, RecursionError):
        return None
    found: set[str] = set()

    def walk(table):
        for sym in table.get_symbols():
            if sym.is_global() and (sym.is_referenced() or sym.is_assigned()):
                found.add(sym.get_name())
        for child in table.get_children():
            walk(child)

    walk(fn_table)
    return found


def function_pairs(source: str, rng: random.Random, max_per_file: int = 2) -> list[dict]:
    """(imports + signature + docstring) -> body, for module-level functions whose body is SELF-CONTAINED.

    Quality rules (a model imitates whatever it is shown, so every one of these removes a bad habit):
      * the function must have a real docstring - it is the only statement of intent in the prompt;
      * every module-level name the body uses must be a builtin, the function itself, or something imported
        by the import lines that are INCLUDED in the prompt. Bodies that read constants/helpers defined
        elsewhere in the original file (EMAIL_REGEX, logger, _helper(...)) would teach the model to invent
        undefined names, so they are dropped;
      * no decorators, no star-imports (names unknown), no one-liners, size limits.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []

    import_texts: list[str] = []
    allowed = set(dir(builtins))
    for node in tree.body:                        # whole import STATEMENTS (multi-line ones stay intact)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno <= 100 and len(import_texts) < 15:
            if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                return []                         # star import: the names it brings are unknown
            seg = ast.get_source_segment(source, node)
            if seg is None:
                continue
            import_texts.append(seg)
            for a in node.names:
                allowed.add((a.asname or a.name).split(".")[0])

    lines = source.splitlines()
    out = []
    for node in tree.body:                        # module-level functions only (no indentation to deal with)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.end_lineno is None or not node.body:
            continue
        if node.decorator_list:
            continue
        first = node.body[0]
        is_doc = (isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant)
                  and isinstance(first.value.value, str))
        if not is_doc or first.lineno == node.lineno or len((ast.get_docstring(node) or "").strip()) < 20:
            continue
        head_end = body_start = first.end_lineno
        header = [l.rstrip() for l in lines[node.lineno - 1:head_end]]
        body = [l.rstrip() for l in lines[body_start:node.end_lineno]]
        while body and not body[-1].strip():
            body.pop()
        if not (2 <= len(body) <= 40) or not any(l.strip() for l in body):
            continue
        header_txt, body_txt = "\n".join(header), "\n".join(body)
        if len(body_txt) > 2000 or len(header_txt) > 1200:
            continue

        if _is_placeholder_body(node.body[1:]):
            continue                              # docstring + pass / ... / raise NotImplementedError teaches "write a stub"
        fn_src = ast.get_source_segment(source, node)
        used = _free_global_names(fn_src) if fn_src else None
        if used is None or (used - allowed - {node.name}):
            continue                              # depends on something the prompt does not show
        ann = [n.id for part in ([a.annotation for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs
                                  if a.annotation] + list(node.args.defaults) + list(node.args.kw_defaults)
                                 + ([node.returns] if node.returns else []))
               if part is not None for n in ast.walk(part) if isinstance(n, ast.Name)]
        if set(ann) - allowed - {node.name}:
            continue                              # an annotation/default names a type or constant defined elsewhere

        prompt = "<file_sep>" + ("\n".join(import_texts) + "\n\n\n" if import_texts else "") + header_txt
        if rng.random() < 0.5:                    # variant A: prompt ends at the last character; response starts with "\n"
            response = "\n" + body_txt + "\n"
        else:                                     # variant B: prompt ends with a newline; response starts at the indentation
            prompt += "\n"
            response = body_txt + "\n"
        out.append({"prompt": prompt, "response": response, "kind": "function"})
    rng.shuffle(out)
    return out[:max_per_file]


# ----------------------------------------------------------------------------
def so_pair(text: str) -> dict | None:
    i = text.find("<answer>")
    if i < 0 or not text.startswith("<question>"):
        return None
    prompt = text[: i + len("<answer>")]
    response = text[i + len("<answer>"):].strip()
    if len(prompt) > 3500 or not (80 <= len(response) <= 3500):
        return None
    return {"prompt": prompt, "response": response, "kind": "stackoverflow"}


def commit_pair(row: dict) -> dict | None:
    """Only SMALL edits (a few changed lines in an otherwise unchanged file): that is the skill to teach."""
    old, new = row.get("old_contents") or "", row.get("new_contents") or ""
    msg = (row.get("message") or row.get("subject") or "").strip()
    if not (5 <= len(msg) <= 300) or old == new:
        return None
    if not (80 <= len(old) <= 3000 and 80 <= len(new) <= 3000):
        return None
    a, b = old.splitlines(), new.splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if sm.ratio() < 0.7:
        return None
    changed = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    if not 1 <= changed <= 12:
        return None
    return {"prompt": f"<commit_before>{old}<commit_message>{msg}<commit_after>", "response": new, "kind": "commit"}


def contest_pairs(row: dict, rng: random.Random, max_per_problem: int = 2) -> list[dict]:
    desc = (row.get("description") or "").strip()
    sols = row.get("solutions") or {}
    py = [c.strip() for lang, c in zip(sols.get("language", []), sols.get("solution", []))
          if lang in bc.PY3 and 30 <= len(c.strip()) <= 3000]
    if not desc or len(desc) > 3500 or not py:
        return []
    return [{"prompt": f"<question>{desc}<answer>", "response": c, "kind": "contest"}
            for c in rng.sample(py, min(max_per_problem, len(py)))]


def apps_pairs(row: dict, rng: random.Random, max_per_problem: int = 2) -> list[dict]:
    try:
        sols = json.loads(row.get("solutions") or "[]")
    except (TypeError, ValueError):
        return []
    sols = [c.strip() for c in sols if isinstance(c, str) and 30 <= len(c.strip()) <= 3000]
    q = (row.get("question") or "").strip()
    if not q or len(q) > 3500 or not sols:
        return []
    return [{"prompt": f"<question>{q}<answer>", "response": c, "kind": "contest"}
            for c in rng.sample(sols, min(max_per_problem, len(sols)))]


# ----------------------------------------------------------------------------
def collect(name: str, local: dict, target: int, make, seed: int, max_scan: int) -> list[dict]:
    """Stream a source until `target` pairs are collected (make(row) -> list of pairs)."""
    if target <= 0:
        return []
    src = bc.BY_NAME[name]
    pairs, scanned = [], 0
    for row in bc.iter_rows(src, local, seed=seed):
        scanned += 1
        try:
            if src.keep is not None and not src.keep(row):
                continue
            pairs.extend(make(row))
        except KeyError:
            continue
        if len(pairs) >= target or scanned >= max_scan:
            break
    print(f"  {name}: {len(pairs):,} pairs from {scanned:,} rows scanned", flush=True)
    return pairs[:target]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="artifacts/sft/sft_pairs.jsonl")
    ap.add_argument("--so-jsonl", help="your stackoverflow.jsonl (from scripts/fetch_stackoverflow.py)")
    ap.add_argument("--n-functions", type=int, default=25_000)
    ap.add_argument("--n-so", type=int, default=15_000)
    ap.add_argument("--n-commits", type=int, default=10_000)
    ap.add_argument("--n-contests", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-scan", type=int, default=3_000_000, help="give up on a source after this many rows")
    ap.add_argument("--local", action="append", help="name=path.jsonl instead of streaming from Hugging Face")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    local = bc.parse_local(args.local)
    pairs: list[dict] = []

    print("function pairs (raw_python)...", flush=True)
    per_file = lambda row: function_pairs(row["content"], rng)  # noqa: E731
    pairs += collect("raw_python", local, args.n_functions, per_file, args.seed, args.max_scan)

    if args.so_jsonl and args.n_so > 0:
        print("stack overflow pairs (reservoir sample of the whole file)...", flush=True)
        reservoir, seen = [], 0
        with open(args.so_jsonl, encoding="utf-8") as f:
            for line in f:
                p = so_pair(json.loads(line)["text"])
                if p is None:
                    continue
                seen += 1
                if len(reservoir) < args.n_so:
                    reservoir.append(p)
                else:
                    j = rng.randrange(seen)
                    if j < args.n_so:
                        reservoir[j] = p
        print(f"  stackoverflow: {len(reservoir):,} pairs sampled from {seen:,} usable", flush=True)
        pairs += reservoir

    print("commit edit pairs...", flush=True)
    pairs += collect("commits", local, args.n_commits, lambda r: [p] if (p := commit_pair(r)) else [], args.seed, args.max_scan)

    print("contest pairs (code_contests + apps)...", flush=True)
    half = args.n_contests * 2 // 3
    pairs += collect("code_contests", local, half, lambda r: contest_pairs(r, rng), args.seed, args.max_scan)
    pairs += collect("apps", local, args.n_contests - half, lambda r: apps_pairs(r, rng), args.seed, args.max_scan)

    # drop exact duplicates, then SHUFFLE (the last 1% becomes the validation split, so the kinds must be mixed)
    seen_h, unique = set(), []
    for p in pairs:
        h = hashlib.md5((p["prompt"] + "\x00" + p["response"]).encode("utf-8", "ignore")).hexdigest()
        if h not in seen_h:
            seen_h.add(h)
            unique.append(p)
    rng.shuffle(unique)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in unique:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    from collections import Counter
    kinds = Counter(p["kind"] for p in unique)
    print(f"\nwrote {len(unique):,} pairs -> {args.out}   (dropped {len(pairs) - len(unique):,} duplicates)")
    for k, n in kinds.most_common():
        sub = [p for p in unique if p["kind"] == k]
        print(f"  {k:<14}{n:>8,}   avg prompt {sum(len(p['prompt']) for p in sub) // n:>5} chars, "
              f"avg response {sum(len(p['response']) for p in sub) // n:>5} chars")
        print(f"      e.g. prompt tail: {sub[0]['prompt'][-70:]!r}\n           response head: {sub[0]['response'][:70]!r}")


if __name__ == "__main__":
    main()
