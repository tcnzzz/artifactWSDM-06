#!/usr/bin/env python3
"""Build number_map.csv: every \\N{key} in the paper -> value, generated file, and the
exact line of code/make_assets.py that produced it.

The paper never types a number by hand.  Each displayed number is a macro
\\N{key} defined in manuscript/generated/numbers.tex, which code/make_assets.py
writes from the saved results.  This script closes the loop: it re-runs
make_assets.py under a line tracer, records the source line that first created
each macro, and joins that with the \\N{key} uses found in the paper.  It exits
non-zero if a key used in the paper has no definition or no source line.

Usage:
    python3 code/number_map.py [--manuscript DIR] [--out FILE] [--no-rerun]

With --no-rerun it skips the tracer and maps only exact/literal assignment
sites, which is faster but leaves loop-generated keys without a line number.
"""
import argparse
import csv
import os
import re
import runpy
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_MS = os.path.join(os.path.dirname(ROOT), "manuscript")
ASSETS = os.path.join(HERE, "make_assets.py")
GEN_FILES = ("numbers.tex", "tab_base.tex", "tab_factors.tex", "tab_repair.tex", "tab_verifier.tex")

DEF_RE = re.compile(r"\\csname N@([^\\]+)\\endcsname\{(.*)\}\s*$")
USE_RE = re.compile(r"\\N\{([^}]*)\}")


def tex_files(ms):
    out = [os.path.join(ms, "main.tex")]
    for sub in ("sections", "generated"):
        d = os.path.join(ms, sub)
        if os.path.isdir(d):
            out += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".tex")]
    return [p for p in out if os.path.exists(p)]


def read_defs(ms):
    """key -> (value, generated file relative to manuscript/)."""
    defs = {}
    for p in tex_files(ms):
        rel = os.path.relpath(p, ms)
        if not rel.startswith("generated" + os.sep):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                m = DEF_RE.search(line.rstrip("\n"))
                if m:
                    defs[m.group(1)] = (m.group(2), rel.replace(os.sep, "/"))
    return defs


def read_uses(ms):
    """key -> list of 'file:line' where the paper uses \\N{key}."""
    uses = {}
    for p in tex_files(ms):
        rel = os.path.relpath(p, ms).replace(os.sep, "/")
        with open(p, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                for m in USE_RE.finditer(line):
                    uses.setdefault(m.group(1), []).append(f"{rel}:{n}")
    return uses


def trace_lines():
    """Run make_assets.py and return key -> (source line, value at creation).

    num is a module-level dict.  A key appears at the next module-level line
    event after its assignment, so attribute it to the previous module line.
    Keys created inside emit() are attributed to the emit(...) call site.
    """
    made = {}
    last_module = [0]

    def tracer(frame, event, arg):
        if event != "line" or os.path.abspath(frame.f_code.co_filename) != ASSETS:
            return tracer
        name = frame.f_code.co_name
        if name == "emit":
            num = frame.f_globals.get("num")
            if isinstance(num, dict):
                for k, v in num.items():
                    if k not in made and frame.f_back is not None:
                        made[k] = (frame.f_back.f_lineno, str(v))
        elif name == "<module>":
            num = frame.f_globals.get("num")
            if isinstance(num, dict):
                for k, v in num.items():
                    if k not in made:
                        made[k] = (last_module[0], str(v))
            last_module[0] = frame.f_lineno
        return tracer

    # Snapshot the files make_assets.py overwrites, and put them back afterwards
    # so that mapping never changes the paper's inputs.
    ms = os.path.join(os.path.dirname(ROOT), "manuscript")
    tmp = tempfile.mkdtemp(prefix="number_map_")
    saved = []
    try:
        for d, names in ((os.path.join(ms, "generated"), GEN_FILES),
                         (os.path.join(ROOT, "results"), ("multiplicity.csv", "multiplicity_table2.csv")),
                         (os.path.join(ms, "figures"), ("fig_dose.pdf",))):
            for name in names:
                src = os.path.join(d, name)
                if os.path.exists(src):
                    dst = os.path.join(tmp, name)
                    shutil.copy2(src, dst)
                    saved.append((src, dst))
        sys.settrace(tracer)
        try:
            runpy.run_path(ASSETS, run_name="__main__")
        finally:
            sys.settrace(None)
    finally:
        for src, dst in saved:
            shutil.copy2(dst, src)
        shutil.rmtree(tmp, ignore_errors=True)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default=DEFAULT_MS)
    ap.add_argument("--out", default=os.path.join(ROOT, "number_map.csv"))
    ap.add_argument("--no-rerun", action="store_true")
    a = ap.parse_args()

    defs = read_defs(a.manuscript)
    uses = read_uses(a.manuscript)
    made = {} if a.no_rerun else trace_lines()

    missing = sorted(k for k in uses if k not in defs)
    rows, untraced = [], []
    for key in sorted(uses):
        value, gen = defs.get(key, ("", ""))
        line, created = made.get(key, ("", ""))
        if created and created != value:
            print(f"WARNING: {key}: generated={value!r} traced={created!r}", file=sys.stderr)
        if line == "":
            untraced.append(key)
        rows.append({
            "key": key,
            "value": value,
            "generated_file": gen,
            "script": "code/make_assets.py",
            "script_line": line,
            "used_in": ";".join(uses[key]),
        })

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["key", "value", "generated_file", "script",
                                          "script_line", "used_in"])
        w.writeheader()
        w.writerows(rows)

    print(f"number_map.csv: {len(rows)} keys used, {len(defs)} defined in generated/, "
          f"{len(made)} traced")
    if missing:
        print(f"MISSING DEFINITIONS ({len(missing)}): " + ", ".join(missing), file=sys.stderr)
    if untraced:
        print(f"UNTRACED KEYS ({len(untraced)}): " + ", ".join(untraced), file=sys.stderr)
    return 1 if (missing or untraced) else 0


if __name__ == "__main__":
    sys.exit(main())
