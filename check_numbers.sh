#!/bin/bash
# check_numbers.sh -- regenerate every number from the saved results and prove that
# the paper's generated/ files and number_map.csv are exactly what the code produces.
#
# It does NOT run a model and does NOT build the PDF.  It rewrites generated/ in
# place for the duration of the diff and restores the shipped files afterwards, so
# a failed check leaves the paper untouched.
#
# Usage:  ./check_numbers.sh [--keep]
# Exit:   0 = every generated file and every macro mapping matches
#         1 = a file differs, a macro is undefined, or a macro has no source line
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT="$HERE"
MS="$HERE/../manuscript"
PY=${PYTHON:-python3}
KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1

fail=0
say() { printf '%s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }

[ -d "$MS/generated" ] || { bad "no manuscript/generated next to the release (expected $MS/generated)"; exit 1; }

if ! "$PY" -c 'import pandas, numpy, matplotlib' >/dev/null 2>&1; then
  say "ERROR: $PY cannot import pandas/numpy/matplotlib."
  say "Create a virtualenv first:  python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
  exit 2
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/check_numbers.XXXXXX")
trap 'if [ "$KEEP" = 0 ]; then cp -a "$TMP/generated/." "$MS/generated/"; [ -f "$TMP/fig_dose.pdf" ] && cp -a "$TMP/fig_dose.pdf" "$MS/figures/fig_dose.pdf"; for f in frontier.csv calibration.csv split_halves.csv duplicate_run.csv feedback_claims.csv; do [ -f "$TMP/results/$f" ] && cp -a "$TMP/results/$f" "$ROOT/results/$f"; done; fi; rm -rf "$TMP"' EXIT

# ---- 1. snapshot the shipped outputs -------------------------------------
mkdir -p "$TMP/generated" "$TMP/results"
cp -a "$MS/generated/." "$TMP/generated/"
for f in frontier.csv calibration.csv split_halves.csv duplicate_run.csv feedback_claims.csv \
         multiplicity.csv multiplicity_table2.csv; do
  cp -a "$ROOT/results/$f" "$TMP/results/$f" 2>/dev/null || true
done
cp -a "$MS/figures/fig_dose.pdf"             "$TMP/fig_dose.pdf"            2>/dev/null || true
cut -d, -f1-7,9- "$TMP/results/frontier.csv" > "$TMP/frontier.notime.csv" 2>/dev/null || true

# ---- 2. regenerate the derived CSVs that make_assets.py reads -------------
say "== regenerating frontier, calibration, robustness and feedback claims from saved results =="
for s in frontier calibration robustness feedback_claims; do
  if ! "$PY" "$ROOT/code/$s.py" > "$TMP/$s.log" 2>&1; then
    cat "$TMP/$s.log"
    bad "code/$s.py exited non-zero"
    exit 1
  fi
  tail -1 "$TMP/$s.log"
done
cut -d, -f1-7,9- "$ROOT/results/frontier.csv" > "$TMP/frontier.new.notime.csv" 2>/dev/null || true
if diff -q "$TMP/frontier.notime.csv" "$TMP/frontier.new.notime.csv" >/dev/null 2>&1; then
  say "results/frontier.csv: identical (time_ms_median excluded)"
else
  bad "results/frontier.csv differs (excluding time_ms_median)"
fi
for f in calibration.csv split_halves.csv duplicate_run.csv feedback_claims.csv; do
  if diff -q "$TMP/results/$f" "$ROOT/results/$f" >/dev/null 2>&1; then
    say "results/$f: identical"
  else
    bad "results/$f differs from the shipped copy"
  fi
done

# ---- 2b. regenerate macros, tables and figures from saved results --------
say "== regenerating macros, tables and figures from saved results =="
if ! "$PY" "$ROOT/code/make_assets.py" > "$TMP/make_assets.log" 2>&1; then
  cat "$TMP/make_assets.log"
  bad "code/make_assets.py exited non-zero"
  exit 1
fi
tail -1 "$TMP/make_assets.log"

# ---- 3. diff generated/ against the shipped copy -------------------------
say "== diffing generated/ =="
if diff -r "$TMP/generated" "$MS/generated" > "$TMP/generated.diff" 2>&1; then
  say "generated/: identical ($(ls "$MS/generated" | tr '\n' ' '))"
else
  bad "generated/ differs from the shipped copy"
  sed -n '1,40p' "$TMP/generated.diff"
fi

# ---- 4. diff the derived multiplicity tables -----------------------------
for f in multiplicity.csv multiplicity_table2.csv; do
  if [ -f "$TMP/results/$f" ] && ! diff -q "$TMP/results/$f" "$ROOT/results/$f" >/dev/null 2>&1; then
    bad "results/$f differs from the shipped copy"
  else
    say "results/$f: identical"
  fi
done

# ---- 5. every \N{key} in the paper has a value and a source line ---------
say "== tracing every \\N{key} to code/make_assets.py =="
if "$PY" "$ROOT/code/number_map.py" --out "$TMP/number_map.csv" > "$TMP/number_map.log" 2>&1; then
  tail -1 "$TMP/number_map.log"
  if ! diff -q "$TMP/number_map.csv" "$ROOT/number_map.csv" >/dev/null 2>&1; then
    bad "number_map.csv is stale (rerun code/number_map.py)"
  else
    say "number_map.csv: identical"
  fi
else
  cat "$TMP/number_map.log"
  bad "a macro is undefined or untraced"
fi

if [ "$fail" = 0 ]; then
  say "ALL CHECKS PASSED"
else
  say "CHECKS FAILED"
fi
exit "$fail"
