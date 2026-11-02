import os
import stat
import sys
import textwrap

import pytest

from mathforge.config import load_config

FAKE_LAKE = textwrap.dedent('''\
    #!{python}
    """Stand-in for `lake env lean FILE` used by the tests.

    Rules: a theorem whose proof contains the tactic `bad_tactic` gets an error on
    that line; a file containing `CRASH` produces no output at all (simulating a
    crash); everything else is accepted and `#print axioms` is answered.
    """
    import re, sys
    path = sys.argv[-1]
    text = open(path, encoding="utf-8").read()
    if "CRASH" in text:
        sys.exit(1)
    lines = text.splitlines()
    bad = set()
    for i, l in enumerate(lines, 1):
        if "bad_tactic" in l:
            print(f"{{path}}:{{i}}:2: error: unknown tactic")
            bad.add(i)
    # find declarations and their line ranges to know which ones failed
    decls = [(i, m.group(1)) for i, l in enumerate(lines, 1)
             for m in [re.match(r"theorem (\\S+)", l)] if m]
    ranges = {{}}
    for k, (i, name) in enumerate(decls):
        end = decls[k + 1][0] - 1 if k + 1 < len(decls) else len(lines)
        ranges[name] = (i, end)
    for l in lines:
        m = re.match(r"#print axioms (\\S+)", l)
        if not m:
            continue
        name = m.group(1)
        lo, hi = ranges.get(name, (0, -1))
        uses_sorry = any("sorry" in lines[j - 1] for j in range(lo, hi + 1))
        failed = any(lo <= j <= hi for j in bad)
        if failed or uses_sorry:
            print(f"'{{name}}' depends on axioms: [propext, sorryAx]")
        else:
            print(f"'{{name}}' depends on axioms: [propext, Classical.choice, Quot.sound]")
    sys.exit(1 if bad else 0)
''')


@pytest.fixture
def fake_lake(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    if os.name == "nt":
        # Windows has no shebang lines: run the script through a lake.cmd wrapper
        script = bindir / "fake_lake.py"
        script.write_text(FAKE_LAKE.format(python=sys.executable), encoding="utf-8")
        (bindir / "lake.cmd").write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        lake = bindir / "lake"
        lake.write_text(FAKE_LAKE.format(python=sys.executable), encoding="utf-8")
        lake.chmod(lake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    proj = tmp_path / "lean_workspace"
    proj.mkdir()
    return str(proj)


@pytest.fixture
def small_cfg(tmp_path):
    def make(*overrides):
        return load_config(None, [f"run.dir={tmp_path / 'run'}", "compute.backend=serial",
                                  "engines.constants.tasks_per_round=2",
                                  "engines.constants.candidates_per_task=60",
                                  "engines.laws.tasks_per_round=2",
                                  "engines.laws.population=150", "engines.laws.generations=15",
                                  *overrides])
    return make
