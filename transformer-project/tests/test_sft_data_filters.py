"""Quality rules for function fine-tuning pairs: bodies must be self-contained."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_sft_data as sd  # noqa: E402


def pairs(src, n=10):
    return sd.function_pairs(src, random.Random(0), max_per_file=n)


DOC = '"""Return something useful computed from the given text value."""'


def test_body_using_a_name_defined_elsewhere_is_dropped():
    src = f"import re\n\nEMAIL_REGEX = re.compile('x')\n\n\ndef f(text):\n    {DOC}\n    out = []\n    for m in EMAIL_REGEX.finditer(text):\n        out.append(m)\n    return out\n"
    assert pairs(src) == []


def test_self_contained_body_is_kept_and_prompt_has_the_import():
    src = f"import re\nimport os\n\n\ndef f(text):\n    {DOC}\n    found = re.findall('x', text)\n    return [m.upper() for m in found if m]\n"
    got = pairs(src)
    assert len(got) == 1
    assert got[0]["prompt"].startswith("<file_sep>import re\nimport os\n\n\ndef f(text):")
    assert "re.findall" in got[0]["response"]


def test_requires_a_real_docstring_no_decorators_no_star_imports():
    body = "    x = [i for i in range(3)]\n    return x\n"
    assert pairs(f"def f(a):\n{body}") == []                                   # no docstring
    assert pairs(f"def f(a):\n    '''short'''\n{body}") == []                  # docstring too short
    assert pairs(f"@staticmethod\ndef f(a):\n    {DOC}\n{body}") == []           # decorated
    assert pairs(f"from os import *\n\n\ndef f(a):\n    {DOC}\n{body}") == []     # unknown names


def test_builtins_params_locals_comprehensions_lambdas_recursion_are_fine():
    src = (f"def fact(n):\n    {DOC}\n    helper = lambda k: k + 1\n    if n <= 1:\n        return 1\n"
           f"    return n * fact(n - 1) + sum(helper(i) for i in range(0))\n")
    assert len(pairs(src)) == 1


def test_multiline_imports_stay_whole_and_names_come_from_them():
    src = (f"from collections import (\n    Counter,\n    defaultdict,\n)\n\n\ndef f(words):\n    {DOC}\n"
           f"    c = Counter(words)\n    d = defaultdict(int)\n    return c, d\n")
    got = pairs(src)
    assert len(got) == 1 and "    Counter,\n    defaultdict,\n)" in got[0]["prompt"]


def test_name_imported_but_not_shown_in_prompt_is_dropped():
    imports = "".join(f"import mod{i}\n" for i in range(16))                  # only the first 15 imports are shown
    src = f"{imports}\n\ndef f(a):\n    {DOC}\n    return mod15.go(a) + 1\n"
    assert pairs(src) == []


def test_annotation_with_undefined_type_is_dropped():
    src = f"def f(a: MyType) -> int:\n    {DOC}\n    return int(a) + 1\n"
    assert pairs(src) == []


def test_stub_bodies_are_dropped():
    for stub in ("    pass\n    pass\n", "    # Your code here\n    pass\n", "    ...\n    ...\n",
                 "    raise NotImplementedError\n    pass\n", "    raise NotImplementedError('todo')\n    pass\n"):
        assert pairs(f"def f(a):\n    {DOC}\n{stub}") == [], stub
    assert len(pairs(f"def f(a):\n    {DOC}\n    # compute it\n    b = a + 1\n    return b\n")) == 1   # a comment + real code is fine
