"""HumanEval/MBPP harness: statistics, completion handling, sandboxed execution, scoring, comparison."""
import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_code_benchmarks as ev  # noqa: E402

HE = [
    {"task_id": "HumanEval/0", "prompt": 'def add(a: int, b: int) -> int:\n    """Return the sum of a and b.\n    >>> add(2, 3)\n    5\n    """\n',
     "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n", "entry_point": "add"},
    {"task_id": "HumanEval/1", "prompt": 'def is_even(n: int) -> bool:\n    """True if n is even."""\n',
     "test": "def check(candidate):\n    assert candidate(4) is True\n    assert candidate(7) is False\n", "entry_point": "is_even"},
    {"task_id": "HumanEval/2", "prompt": 'def rev(s: str) -> str:\n    """Reverse the string."""\n',
     "test": "def check(candidate):\n    assert candidate('abc') == 'cba'\n    assert candidate('') == ''\n", "entry_point": "rev"},
]
GOOD = {"HumanEval/0": "    return a + b\n", "HumanEval/1": "    return n % 2 == 0\n", "HumanEval/2": "    return s[::-1]\n"}
MBPP = [{"task_id": 11, "text": "Write a function to double a number.", "test_list": ["assert double(2) == 4", "assert double(0) == 0", "assert double(-3) == -6"], "test_setup_code": ""}]


def test_pass_at_k_matches_known_values():
    assert ev.pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert ev.pass_at_k(10, 0, 5) == 0.0
    assert ev.pass_at_k(10, 3, 10) == 1.0                      # one of all ten must be a passing one
    assert ev.pass_at_k(5, 1, 2) == pytest.approx(1 - (4 / 5) * (3 / 4))   # 1 - C(4,2)/C(5,2) = 0.4


def test_prompt_completion_and_stops():
    full, raw = ev.build_prompt(HE[0], "humaneval", file_sep=True)
    assert full == "<file_sep>" + raw
    generated = raw + "    return a + b\n\n\ndef other():\n    pass\n"           # decoder hides <file_sep>, so the echo starts at raw
    comp = ev.truncate_at_stops(ev.extract_completion(generated, raw), ev.HE_STOPS)
    assert comp.rstrip() == "    return a + b"                          # trailing blank lines are harmless
    assert ev.truncate_at_stops("x = 1\nassert x\nmore", ev.MBPP_STOPS) == "x = 1"
    _, mraw = ev.build_prompt(MBPP[0], "mbpp")
    assert mraw == '"""\nWrite a function to double a number.\nassert double(2) == 4\n"""\n'


def test_sandbox_passes_fails_times_out_and_blocks_damage(tmp_path):
    assert ev.run_program("assert 1 + 1 == 2\n")[0] is True
    ok, reason = ev.run_program("assert 1 + 1 == 3\n")
    assert not ok and ev.error_class(reason) == "AssertionError"
    ok, reason = ev.run_program("def f(:\n", timeout=5)
    assert not ok and ev.error_class(reason) == "SyntaxError"
    ok, reason = ev.run_program("while True:\n    pass\n", timeout=2)
    assert not ok and reason == "timeout"
    ok, reason = ev.run_program("x = bytearray(2 * 10**9)\n", timeout=10, mem_mb=512)
    assert not ok and "MemoryError" in reason
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    prog = f"import os\nos.remove({str(victim)!r})\n"
    assert not ev.run_program(prog)[0] and victim.exists()                      # destructive call is disabled
    prog = f"import os\nos.system('rm -f {victim}')\n"
    assert not ev.run_program(prog)[0] and victim.exists()
    prog = f"import shutil\nshutil.rmtree({str(tmp_path)!r})\n"
    assert not ev.run_program(prog)[0] and victim.exists()


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _score(tmp_path, comps, name, benchmark="humaneval", data=HE):
    _write(tmp_path / "data.jsonl", data)
    _write(tmp_path / f"{name}.comp.jsonl", comps)
    out = tmp_path / f"{name}.results.jsonl"
    ev.cmd_score(argparse.Namespace(benchmark=benchmark, data=str(tmp_path / "data.jsonl"), completions=str(tmp_path / f"{name}.comp.jsonl"),
                                    out=str(out), workers=2, timeout=5.0, mem_mb=1024))
    return [json.loads(l) for l in open(out)]


def test_score_and_compare_end_to_end(tmp_path, capsys):
    base = [{"task_id": t, "sample": 0, "completion": c} for t, c in GOOD.items()]
    base[1]["completion"] = "    return n % 2 == 1\n"                           # wrong
    base[2]["completion"] = "    pass\n"                                        # wrong
    tuned = [{"task_id": t, "sample": 0, "completion": c} for t, c in GOOD.items()]   # all correct
    rb = _score(tmp_path, base, "base")
    rt = _score(tmp_path, tuned, "tuned")
    assert [r["passed"] for r in rb] == [True, False, False]
    assert all(r["passed"] for r in rt)
    out = capsys.readouterr().out
    assert "pass@1  = 33.3%" in out and "pass@1  = 100.0%" in out
    ev.cmd_compare(argparse.Namespace(a=str(tmp_path / "base.results.jsonl"), b=str(tmp_path / "tuned.results.jsonl")))
    out = capsys.readouterr().out
    assert "B - A = +66.7 points" in out and "problems where B does better: 2" in out


def test_score_mbpp_and_pass_at_k_with_several_samples(tmp_path, capsys):
    comps = [{"task_id": "11", "sample": i, "completion": c} for i, c in enumerate(
        ["def double(x):\n    return x * 2\n", "def double(x):\n    return x + 2\n"] * 5)]
    res = _score(tmp_path, comps, "m", benchmark="mbpp", data=MBPP)
    assert sum(r["passed"] for r in res) == 5
    out = capsys.readouterr().out
    assert "1 problems x 10 sample(s)" in out and "pass@1  = 50.0%" in out and "pass@10 = 100.0%" in out


def test_unknown_task_ids_are_rejected(tmp_path):
    with pytest.raises(SystemExit):
        _score(tmp_path, [{"task_id": "nope", "sample": 0, "completion": "x"}], "bad")
