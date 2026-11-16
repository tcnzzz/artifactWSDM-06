#!/bin/bash
# reproduce.sh -- rebuild every number in the paper from the saved model outputs,
# rebuild the PDF, and check that the result matches the shipped files.
#
# It runs no model and makes no network call.  Inputs are results/*.csv|json|jsonl,
# data/*.json and results/raw/*.jsonl; the only thing executed is Python + latexmk.
#
# Usage:
#   ./reproduce.sh            # regenerate, compare, build, verify
#   ./reproduce.sh --no-build # regenerate and compare only
#
# The paper source must sit next to this directory as ../manuscript (the release
# directory is normally named artifact/).  Exit 0 means: numbers reproduced
# byte-for-byte, PDF built, no ?? and no undefined reference.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT="$HERE"
MS="$HERE/../manuscript"
PY=${PYTHON:-python3}
BUILD=1
[ "${1:-}" = "--no-build" ] && BUILD=0

fail=0
say() { printf '%s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }

say "release: $ROOT"
say "paper:   $MS"
[ -d "$MS" ] || { bad "no manuscript/ next to the release"; exit 1; }
[ -d "$MS/generated" ] || { bad "no manuscript/generated"; exit 1; }

if ! "$PY" -c 'import pandas, numpy, matplotlib' >/dev/null 2>&1; then
  say "ERROR: $PY cannot import pandas/numpy/matplotlib."
  say "Create a virtualenv first:  python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
  exit 2
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/reproduce.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

# ---- 1. snapshot the shipped numbers -------------------------------------
mkdir -p "$TMP/generated"
cp -a "$MS/generated/." "$TMP/generated/"
for f in frontier.csv multiplicity.csv multiplicity_table2.csv slack_strata.csv \
         position_sensitivity.csv calibration.csv split_halves.csv duplicate_run.csv \
         feedback_claims.csv; do
  cp -a "$ROOT/results/$f" "$TMP/$f" 2>/dev/null || true
done
cp -a "$MS/figures/fig_dose_pgf.tex" "$TMP/fig_dose_pgf.tex" 2>/dev/null || true
cp -a "$MS/figures/data/fig_dose_series.json" "$TMP/fig_dose_series.json" 2>/dev/null || true
cp -a "$ROOT/number_map.csv" "$TMP/number_map.csv" 2>/dev/null || true
# frontier.csv carries a wall-clock timing column, which is reported but not
# part of the byte-identical check (the shipped copy is from another machine).
cut -d, -f1-7,9- "$ROOT/results/frontier.csv" > "$TMP/frontier.notime.csv" 2>/dev/null || true

# ---- 2. regenerate from saved outputs ------------------------------------
say "== 1/11 max-min slack strata =="
"$PY" "$ROOT/code/slack_strata.py" 2>&1 | tee "$TMP/slack_strata.log"
[ "${PIPESTATUS[0]}" = 0 ] || { bad "slack_strata.py failed"; exit 1; }

say "== 2/11 position-weight sensitivity =="
"$PY" "$ROOT/code/position_sensitivity.py" 2>&1 | tee "$TMP/position.log"
[ "${PIPESTATUS[0]}" = 0 ] || bad "position_sensitivity.py failed"

say "== 3/11 nDCG-violation frontier and baseline heuristics =="
"$PY" "$ROOT/code/frontier.py" 2>&1 | tee "$TMP/frontier.log"
[ "${PIPESTATUS[0]}" = 0 ] || bad "frontier.py failed"

say "== 4/11 calibration of the token probability =="
"$PY" "$ROOT/code/calibration.py" 2>&1 | tee "$TMP/calibration.log"
[ "${PIPESTATUS[0]}" = 0 ] || bad "calibration.py failed"

say "== 5/11 split-half and duplicate-run robustness =="
"$PY" "$ROOT/code/robustness.py" 2>&1 | tee "$TMP/robustness.log"
[ "${PIPESTATUS[0]}" = 0 ] || bad "robustness.py failed"

say "== 6/11 feedback-loop claim honesty =="
"$PY" "$ROOT/code/feedback_claims.py" 2>&1 | tee "$TMP/feedback_claims.log"
[ "${PIPESTATUS[0]}" = 0 ] || bad "feedback_claims.py failed"

say "== 7/11 reporting checks and clustered intervals =="
"$PY" "$ROOT/code/verify_reporting_checks.py" --artifact "$ROOT" 2>&1 | tee "$TMP/verify_reporting.log"
[ "${PIPESTATUS[0]}" = 0 ] || bad "verify_reporting_checks.py failed"
"$PY" "$ROOT/code/verify_writing_revision.py" --artifact "$ROOT" 2>&1 | tee "$TMP/verify_writing.log"
[ "${PIPESTATUS[0]}" = 0 ] || bad "verify_writing_revision.py failed"

say "== 8/11 macros, tables and dose figure =="
"$PY" "$ROOT/code/make_assets.py" 2>&1 | tee "$TMP/make_assets.log"
[ "${PIPESTATUS[0]}" = 0 ] || { bad "make_assets.py failed"; exit 1; }

say "== 9/11 number map =="
"$PY" "$ROOT/code/number_map.py" 2>&1 | tee "$TMP/number_map.log"
[ "${PIPESTATUS[0]}" = 0 ] || bad "number_map.py failed"

# ---- 3. compare with the shipped copies ---------------------------------
say "== 10/11 diff against shipped files =="
if diff -r "$TMP/generated" "$MS/generated" > "$TMP/generated.diff" 2>&1; then
  say "generated/: identical"
else
  bad "generated/ changed"
  sed -n '1,40p' "$TMP/generated.diff"
fi
for pair in "fig_dose_pgf.tex:$MS/figures/fig_dose_pgf.tex" \
            "fig_dose_series.json:$MS/figures/data/fig_dose_series.json"; do
  name=${pair%%:*}; dest=${pair#*:}
  if diff -q "$TMP/$name" "$dest" >/dev/null 2>&1; then
    say "figures/$name: identical"
  else
    bad "figures/$name changed"
  fi
done
cut -d, -f1-7,9- "$ROOT/results/frontier.csv" > "$TMP/frontier.new.notime.csv" 2>/dev/null || true
if diff -q "$TMP/frontier.notime.csv" "$TMP/frontier.new.notime.csv" >/dev/null 2>&1; then
  say "frontier.csv: identical (time_ms_median excluded)"
else
  bad "frontier.csv changed (excluding time_ms_median)"
fi
for f in multiplicity.csv multiplicity_table2.csv slack_strata.csv position_sensitivity.csv \
         calibration.csv split_halves.csv duplicate_run.csv feedback_claims.csv number_map.csv; do
  [ -f "$TMP/$f" ] || continue
  case "$f" in
    number_map.csv) shipped="$ROOT/$f" ;;
    *)              shipped="$ROOT/results/$f" ;;
  esac
  if diff -q "$TMP/$f" "$shipped" >/dev/null 2>&1; then
    say "$f: identical"
  else
    bad "$f changed"
  fi
done

# ---- 4. build the paper --------------------------------------------------
if [ "$BUILD" = 1 ]; then
  if ! command -v latexmk >/dev/null 2>&1; then
    bad "latexmk not found; rerun with --no-build to skip the PDF"
  else
    say "== 11/11 building the PDF =="
    ( cd "$MS" && latexmk -C >/dev/null 2>&1
      latexmk -pdf -interaction=nonstopmode main.tex > "$TMP/build.log" 2>&1 )
    if [ ! -f "$MS/main.pdf" ]; then
      bad "main.pdf was not produced"; tail -30 "$TMP/build.log"
    else
      if command -v pdfinfo >/dev/null 2>&1; then
        say "$(pdfinfo "$MS/main.pdf" | grep -E '^Pages:')"
      fi
      if command -v pdftotext >/dev/null 2>&1; then
        if pdftotext "$MS/main.pdf" - | grep -q '??'; then
          bad "the PDF contains ?? (a macro is undefined)"
        else
          say "no ?? in the PDF text"
        fi
      fi
      if grep -qiE 'Citation.*undefined|Reference.*undefined' "$MS/main.log"; then
        bad "undefined citation or reference"
      else
        say "no undefined citations or references"
      fi
      grep -n 'Overfull' "$MS/main.log" | head -3
    fi
  fi
fi

if [ "$fail" = 0 ]; then
  say "REPRODUCED: numbers match and the paper builds"
else
  say "REPRODUCTION FAILED"
fi
exit "$fail"
