"""
HumanEval and MBPP benchmarks for your model.

  HumanEval  164 hand-written Python problems. The model gets a function signature + docstring and
             writes the body; hidden unit tests decide pass/fail.
  MBPP       500 test problems ("write a function that ..."); the model sees the description and the first
             test, and its code must pass 3 asserts.

Three steps (each its own subcommand):

  generate   run YOUR model on the problems  -> completions.jsonl          (needs a GPU; uses the project code)
  score      run every completion against the tests in a sandboxed subprocess -> results.jsonl + a summary
             with pass@1 (and pass@5/@10 if you sampled enough) and a 95% confidence interval
  compare    paired comparison of two results files (e.g. base vs fine-tuned): difference in points + 95% CI

Typical use:
  python scripts/eval_code_benchmarks.py generate --config configs/run_sft.yaml --checkpoint artifacts/base/base.pt \\
      --benchmark humaneval --out artifacts/eval/he_base.jsonl --n 1 --temperature 0
  python scripts/eval_code_benchmarks.py score --benchmark humaneval --completions artifacts/eval/he_base.jsonl \\
      --out artifacts/eval/he_base.results.jsonl

What the numbers mean
  pass@1  = chance a single answer passes (with --n 1 --temperature 0 it is just the fraction of problems solved)
  pass@k  = chance that at least one of k answers passes (needs --n >= k samples at temperature > 0)
  The 95% CI comes from resampling the PROBLEMS: with 164 problems it is typically +/- 5-6 points, which is why
  30 samples of one prompt told you nothing.

Safety: generated code is run in a separate process with a time limit, a memory limit, a throw-away working
directory, a minimal environment and common destructive functions disabled. This is NOT a security sandbox -
run it on a disposable machine such as a Colab runtime, never on a computer with data you care about.

Contamination: HumanEval/MBPP problems and solutions are widely copied on GitHub, so a model trained on a large
GitHub crawl may have seen them. Treat absolute numbers with some caution; the base-vs-fine-tuned COMPARISON on
the same problems is still meaningful.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HE_STOPS = ["\nclass", "\ndef", "\n#", "\nif", "\nprint", "\n@"]
MBPP_STOPS = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"]

HF_SOURCES = {
    "humaneval": [("openai/openai_humaneval", None, "test"), ("openai_humaneval", None, "test")],
    "mbpp": [("google-research-datasets/mbpp", "full", "test"), ("mbpp", "full", "test")],
}


# ----------------------------------------------------------------------------
# problems, prompts, completions
# ----------------------------------------------------------------------------
def load_problems(benchmark: str, data_path: str | None, limit: int = 0) -> list[dict]:
    if data_path:
        problems = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    else:
        from datasets import load_dataset  # lazy: only needed when downloading

        last = None
        for name, config, split in HF_SOURCES[benchmark]:
            try:
                ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
                problems = [dict(r) for r in ds]
                break
            except Exception as e:  # noqa: BLE001 - try the next known id
                last = e
        else:
            raise SystemExit(f"Could not download {benchmark} from the Hub ({last}). Give a local jsonl with --data.")
    for p in problems:
        p["task_id"] = str(p["task_id"])
    return problems[:limit] if limit else problems


def build_prompt(problem: dict, benchmark: str, file_sep: bool = True) -> tuple[str, str]:
    """Returns (text fed to the model, the same text without the leading <file_sep> tag)."""
    if benchmark == "humaneval":
        raw = problem["prompt"]
    else:   # MBPP, in the form used by the BigCode evaluation harness: description + first test inside a docstring
        raw = f'"""\n{problem["text"]}\n{problem["test_list"][0]}\n"""\n'
    return ("<file_sep>" if file_sep else "") + raw, raw


def extract_completion(full_text: str, raw_prompt: str) -> str:
    """The generator returns prompt + continuation (special tags are hidden by the decoder): cut the prompt off."""
    if full_text.startswith(raw_prompt):
        return full_text[len(raw_prompt):]
    tail = raw_prompt[-60:]
    k = full_text.find(tail)
    return full_text[k + len(tail):] if k >= 0 else full_text


def truncate_at_stops(text: str, stops: list[str]) -> str:
    cut = [text.find(s) for s in stops if s in text]
    return text[: min(cut)] if cut else text


def build_program(problem: dict, benchmark: str, completion: str) -> str:
    _, raw = build_prompt(problem, benchmark, file_sep=False)
    if benchmark == "humaneval":
        return raw + completion + "\n\n" + problem["test"] + "\n\n" + f"check({problem['entry_point']})\n"
    setup = problem.get("test_setup_code") or ""
    return raw + completion + "\n\n" + setup + "\n" + "\n".join(problem["test_list"]) + "\n"


# ----------------------------------------------------------------------------
# sandboxed execution
# ----------------------------------------------------------------------------
GUARD = '''
import builtins as _b, os as _os, shutil as _sh, subprocess as _sp
for _n in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork", "forkpty", "killpg",
           "rename", "renames", "truncate", "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
           "lchown", "chdir", "popen"):
    if hasattr(_os, _n):
        setattr(_os, _n, None)
_sh.rmtree = _sh.move = _sh.chown = None
_sp.Popen = _sp.run = _sp.call = _sp.check_call = _sp.check_output = None
_b.exit = _b.quit = None
del _b, _os, _sh, _sp
'''

# runs inside the child process: set limits, then execute the program file
LAUNCHER = (
    "import resource, sys\n"
    "path, mem, cpu = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])\n"
    "resource.setrlimit(resource.RLIMIT_AS, (mem, mem))\n"
    "resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))\n"
    "resource.setrlimit(resource.RLIMIT_FSIZE, (10 << 20, 10 << 20))\n"
    "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
    "src = open(path).read()\n"
    "exec(compile(src, path, 'exec'), {'__name__': '__main__'})\n"
)


def run_program(program: str, timeout: float = 10.0, mem_mb: int = 4096) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        Path(path).write_text(GUARD + program, encoding="utf-8")
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
        proc = subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", LAUNCHER, path, str(mem_mb << 20), str(int(timeout) + 2)],
            cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
            return False, "timeout"
        if proc.returncode == 0:
            return True, "ok"
        lines = [l for l in (err or "").strip().splitlines() if l.strip()]
        return False, (lines[-1][:200] if lines else f"exit code {proc.returncode}")


def error_class(reason: str) -> str:
    m = re.match(r"^([A-Za-z_]*(?:Error|Exception|Warning|Interrupt))", reason)
    return m.group(1) if m else ("timeout" if reason == "timeout" else "other")


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of 'at least one of k samples passes' from n samples with c passing (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def bootstrap_ci(values, iters: int = 4000, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(iters, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_problem(results: list[dict]) -> dict[str, tuple[int, int]]:
    agg: dict[str, list[int]] = {}
    for r in results:
        a = agg.setdefault(r["task_id"], [0, 0])
        a[0] += 1
        a[1] += int(r["passed"])
    return {k: (v[0], v[1]) for k, v in agg.items()}


def summarize(results: list[dict], title: str = "") -> dict:
    pp = per_problem(results)
    ns = {n for n, _ in pp.values()}
    n = min(ns)
    p1 = [c / m for m, c in pp.values()]
    lo, hi = bootstrap_ci(p1)
    out = {"problems": len(pp), "samples_per_problem": n, "pass@1": float(np.mean(p1)), "ci95": (lo, hi)}
    for k in (5, 10):
        if n >= k:
            out[f"pass@{k}"] = float(np.mean([pass_at_k(m, c, k) for m, c in pp.values()]))
    print(f"{title}{len(pp)} problems x {n} sample(s)")
    print(f"  pass@1  = {out['pass@1']:.1%}   (95% CI {lo:.1%} - {hi:.1%})")
    for k in (5, 10):
        if f"pass@{k}" in out:
            print(f"  pass@{k:<2} = {out[f'pass@{k}']:.1%}")
    return out


# ----------------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------------
def cmd_generate(args) -> None:
    import torch
    from transformer_project.config import RunConfig
    from transformer_project.data.tokenizer import TokenizerWrapper
    from transformer_project.inference.generate import generate
    from transformer_project.model.transformer import Transformer
    from transformer_project.training.checkpoint import load_checkpoint

    problems = load_problems(args.benchmark, args.data, args.limit)
    stops = HE_STOPS if args.benchmark == "humaneval" else MBPP_STOPS
    greedy = args.temperature <= 0
    n = 1 if greedy else args.n
    if greedy and args.n > 1:
        print("temperature 0 = greedy: every sample would be identical, so using n=1")

    cfg = RunConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = TokenizerWrapper.load(cfg.data.tokenizer_path)
    model = Transformer(cfg.model)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.to(device).eval()
    torch.manual_seed(args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        for i, p in enumerate(problems):
            full_prompt, raw = build_prompt(p, args.benchmark, file_sep=not args.no_file_sep)
            for s in range(n):
                text = generate(
                    model, tokenizer, full_prompt, device, max_new_tokens=args.max_new_tokens,
                    strategy="greedy" if greedy else "sample", temperature=max(args.temperature, 1e-6),
                    top_k=args.top_k, top_p=args.top_p, repetition_penalty=1.0, stop_strings=stops,
                    use_amp=device.type == "cuda", amp_dtype=torch.bfloat16)
                completion = truncate_at_stops(extract_completion(text, raw), stops)
                f.write(json.dumps({"task_id": p["task_id"], "sample": s, "completion": completion}) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(problems):
                el = time.time() - t0
                print(f"  {i + 1}/{len(problems)} problems | {el / 60:.1f} min | ~{el / (i + 1) * (len(problems) - i - 1) / 60:.1f} min left", flush=True)
    print(f"wrote {args.out}")


def cmd_score(args) -> None:
    problems = {p["task_id"]: p for p in load_problems(args.benchmark, args.data)}
    comps = [json.loads(l) for l in open(args.completions, encoding="utf-8") if l.strip()]
    unknown = {c["task_id"] for c in comps} - set(problems)
    if unknown:
        raise SystemExit(f"{len(unknown)} completion task_ids are not in the {args.benchmark} data, e.g. {sorted(unknown)[:3]}")

    def job(c):
        ok, reason = run_program(build_program(problems[c["task_id"]], args.benchmark, c["completion"]),
                                 timeout=args.timeout, mem_mb=args.mem_mb)
        return {"task_id": c["task_id"], "sample": c.get("sample", 0), "passed": ok, "reason": reason}

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(job, comps))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"ran {len(results)} programs in {time.time() - t0:.0f}s -> {args.out}")
    summarize(results, f"{args.benchmark}: ")
    errs = Counter(error_class(r["reason"]) for r in results if not r["passed"])
    if errs:
        print("  why the failures failed:", ", ".join(f"{k} {v}" for k, v in errs.most_common(6)))


def cmd_compare(args) -> None:
    a = per_problem([json.loads(l) for l in open(args.a, encoding="utf-8") if l.strip()])
    b = per_problem([json.loads(l) for l in open(args.b, encoding="utf-8") if l.strip()])
    ids = sorted(set(a) & set(b))
    if not ids:
        raise SystemExit("the two files have no problems in common")
    pa = np.array([a[i][1] / a[i][0] for i in ids])
    pb = np.array([b[i][1] / b[i][0] for i in ids])
    d = pb - pa
    lo, hi = bootstrap_ci(d)
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    print(f"{len(ids)} problems in common")
    print(f"  A: {args.a}   pass@1 {pa.mean():.1%}")
    print(f"  B: {args.b}   pass@1 {pb.mean():.1%}")
    print(f"  B - A = {d.mean() * 100:+.1f} points   (95% CI {lo * 100:+.1f} to {hi * 100:+.1f})")
    print(f"  problems where B does better: {wins} | worse: {losses} | same: {len(ids) - wins - losses}")
    verdict = ("B is better" if lo > 0 else "A is better" if hi < 0 else "no clear difference (the interval includes 0)")
    print(f"  verdict: {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--config", required=True)
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--benchmark", choices=["humaneval", "mbpp"], required=True)
    g.add_argument("--data", help="local jsonl instead of downloading from the Hub")
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=1, help="samples per problem (needs temperature > 0 for n > 1)")
    g.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--max-new-tokens", type=int, default=300)
    g.add_argument("--limit", type=int, default=0, help="only the first L problems (quick check)")
    g.add_argument("--no-file-sep", action="store_true", help="do not prepend the <file_sep> tag the model saw in pretraining")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_generate)

    s = sub.add_parser("score")
    s.add_argument("--benchmark", choices=["humaneval", "mbpp"], required=True)
    s.add_argument("--data")
    s.add_argument("--completions", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=max(2, min(8, (os.cpu_count() or 2) * 2)))
    s.add_argument("--timeout", type=float, default=10.0)
    s.add_argument("--mem-mb", type=int, default=4096)
    s.set_defaults(fn=cmd_score)

    c = sub.add_parser("compare")
    c.add_argument("a", help="results.jsonl of model A (e.g. base)")
    c.add_argument("b", help="results.jsonl of model B (e.g. fine-tuned)")
    c.set_defaults(fn=cmd_compare)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
