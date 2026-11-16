#!/usr/bin/env python3
"""Recompute the clustered token-probability intervals and one-sided rates.

Usage:  python3 code/verify_writing_revision.py [--artifact DIR]

Reads saved records only.  Exits non-zero if the point estimates, the
2,000-replicate intervals or the one-sided/all-floor rates change.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reporting_checks as rc  # noqa: E402
import writing_revision as wr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=os.path.dirname(HERE))
    a = ap.parse_args()
    root = a.artifact
    rows = [json.loads(line) for line in
            open(os.path.join(root, "results", "checked.jsonl"), encoding="utf-8")]
    probs = rc.load_probs(root)

    ci = wr.clustered_ci(rows, probs, rc.MODELS)
    expect = {
        "qwen2.5:7b": (3980, 0.538886, 0.5092, 0.5689),
        "llama3.1:8b": (3981, 0.430068, 0.3836, 0.4764),
        "gemma2:9b": (3999, 0.384382, 0.3487, 0.4198),
    }
    for x in ci:
        n, auroc, lo, hi = expect[x["model"]]
        assert x["n"] == n, (x["model"], x["n"], n)
        assert abs(x["auroc"] - auroc) < 1e-6, (x["model"], x["auroc"], auroc)
        assert abs(x["ci_low"] - lo) < 1e-4 and abs(x["ci_high"] - hi) < 1e-4, (x["model"], x["ci_low"], x["ci_high"])

    one = wr.one_sided_rates(rows)
    expect_one = {"qwen2.5:7b": (103, 129), "llama3.1:8b": (180, 189), "gemma2:9b": (107, 147)}
    for x in one:
        assert x["n"] == 189
        assert (x["one_sided"], x["all_groups"]) == expect_one[x["model"]], (x["model"], x)

    print("writing revision: clustered CIs and one-sided/all-floor counts OK")


if __name__ == "__main__":
    main()
