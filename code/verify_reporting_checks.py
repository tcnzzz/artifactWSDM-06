#!/usr/bin/env python3
"""Recompute the paper's reporting checks from saved records and assert them.

Usage:  python3 code/verify_reporting_checks.py [--artifact DIR]

Only saved files are read: results/checked.jsonl, results/ptrue/*.jsonl and
results/feedback_claims.csv.  The script prints a short summary and exits
non-zero if a published headline population changes.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reporting_checks as rc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=os.path.dirname(HERE))
    a = ap.parse_args()
    root = a.artifact
    rows = [json.loads(line) for line in
            open(os.path.join(root, "results", "checked.jsonl"), encoding="utf-8")]
    probs = rc.load_probs(root)
    with open(os.path.join(root, "results", "feedback_claims.csv"), encoding="utf-8") as f:
        feedback = list(csv.DictReader(f))
    r = rc.compute(rows, probs, feedback)

    core, q3 = r["rounding"]
    assert core["exact"]["n"] == 10038
    assert core["exact"]["violations"] == 6552
    assert core["exact"]["false_assurances"] == 6552
    assert core["displayed"]["violations"] == 6544
    assert core["stable"]["n"] == 10018 and core["stable"]["violations"] == 6538
    assert core["changed"] == 20
    assert core["unique_slates"] == 477
    assert q3["exact"]["violations"] == 14 and q3["exact"]["caught"] == 2
    assert q3["exact"]["false_assurances"] == 12 and q3["exact"]["claims_compliant"] == 96
    assert q3["displayed"]["violations"] == 5 and q3["displayed"]["caught"] == 2
    assert q3["displayed"]["false_assurances"] == 3
    assert q3["stable"]["n"] == 89 and q3["changed"] == 9
    assert r["infeasible_core_G5"] == {"n": 276, "claims_compliant": 276}

    ml1m = {(x["model"], x["dataset"]): x for x in r["matched_audits"]}
    for model, (viol, caught, alarms, compliant, false, claims) in {
        "qwen2.5:7b": (155, 125, 26, 34, 30, 38),
        "llama3.1:8b": (182, 169, 6, 7, 13, 14),
        "gemma2:9b": (160, 27, 0, 29, 133, 162),
        "qwen2.5:14b": (149, 87, 10, 40, 62, 92),
    }.items():
        z = ml1m[(model, "ml1m")]["audit"]
        assert (z["violations"], z["caught"], z["false_alarms"], z["compliant"],
                z["false_assurances"], z["claims_compliant"]) == (viol, caught, alarms, compliant, false, claims)

    qf = next(x for x in r["feedback"] if x["model"] == "qwen2.5:7b"
              and x["cell"] == "feedback_retry_final")
    assert (qf["valid"], qf["violations"], qf["caught"], qf["false_alarms"],
            qf["false_assurances"], qf["claims_compliant"]) == (188, 128, 80, 2, 48, 106)
    assert abs(qf["boolean_auroc"] - 0.7958333333333334) < 1e-12

    print("reporting checks: 10038/6552/6544, Qwen3 14/5/2, audits, feedback OK")


if __name__ == "__main__":
    main()
