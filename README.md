# Artifact: do LLM rerankers know when they miss a provider-exposure floor?

Everything displayed in the manuscript is generated from saved model outputs by the scripts
below. No number is typed by hand.

## Overview

The artifact contains the saved outputs of seven instruction-tuned models (Qwen2.5-7B,
Llama-3.1-8B, Gemma-2-9B, Qwen2.5-14B, Command-R-35B, Llama-3.3-70B, Qwen3-8B) on slates from
MovieLens-1M and MIND-small, the scripts that turn those outputs into every number in the paper,
and the exact-repair reference solver. Verifying the paper requires no model call and no network
request: `check_numbers.sh` rebuilds `generated/` from the saved outputs and diffs it against the
shipped copy, and `reproduce.sh` additionally rebuilds the PDF.

The reference points are a weighted-DCG integer program solved with SciPy's `milp` (SciPy 1.18.1,
bundling HiGHS 1.12.0); `SOLVER.md` documents the variables, objective, constraints, defaults,
problem sizes, infeasible-slate handling and the timing protocol, including the hardware caveat
for the reported median.

The paper source normally sits next to this directory as `../manuscript`; see "Verify without
running a model" below. If it is absent, both scripts fall back to the `generated/` and
`figures/` copies shipped here: every number is still regenerated from the saved results and
diffed, and only the two checks that read the `.tex` sources (the macro-usage map and the PDF
build) are skipped.

## Layout

```text
code/select_recommender.py  validation comparison of candidate generators for MovieLens-1M
code/prep_data.py           data/slates_ml1m.json (item-kNN) and the first MIND version
code/prep_mind_v2.py        data/slates_mind.json: MIND scores blended with popularity (first version kept)
code/prep_history.py        data/history_*.json: the user's ten most recent items for the history prompt
code/fairlib.py             exposure, constraint check, exact integer-program repair, FA*IR wrapper
code/llm_rerank.py          reranking + claim elicitation for every cell, one JSONL line per output
code/llm_feedback.py        numeric-feedback retry and violation-memory (FACTER-style) loops
code/run_local.sh           Qwen sweep on the Ollama server (core cells)
code/run_queue.sh           prioritised sweep per model on a local llama.cpp server
code/run_server_queue.sh    remaining Llama/Gemma ML-1M cells on llama.cpp CUDA (remote server, ports 1157x)
code/run_server_qwen.sh     Qwen ML-1M G3/G5 (and MIND cells of the NaN version) on llama.cpp CUDA
code/run_mind_queue.sh      all MIND cells per model on llama.cpp CUDA
code/traces.py              stated vs true shares in reasoning traces -> results/traces.csv
code/fidelity.py            Ollama vs llama.cpp agreement on shared slates
code/ptrue.py               token probability of the claim by teacher forcing
code/analyze.py             numeric check of every output -> results/checked.jsonl, results/summary.csv
code/reporting_checks.py    rounding, infeasible-slate, audit and feedback checks -> used by make_assets.py
code/writing_revision.py    clustered token-probability intervals and one-sided/all-floor rates
code/direction.py           which way the under-exposed group moves -> results/direction.csv
code/verifier.py            acceptance rules under a synthetic hidden-patience factor -> results/verifier*.csv
code/sim_crc.py             numeric check of Theorem 1 -> results/sim_crc.csv
code/frontier.py            nDCG-violation frontier plus B7 baseline heuristics -> results/frontier.csv, figures/fig_frontier.pdf
code/calibration.py         B8 Brier score and 10-bin ECE of P(True) -> results/calibration.csv
code/robustness.py          B9 split-half check and Qwen3 partial duplicate run -> results/split_halves.csv, results/duplicate_run.csv
code/feedback_claims.py     B10 final-round claim honesty in the retry/memory loops -> results/feedback_claims.csv
code/slack_strata.py        slate-level max-min slack t* (HiGHS) -> results/slack_strata.csv
code/position_sensitivity.py  post-hoc re-scoring under inverse/linear/top-5/geometric weights
code/make_assets.py         macros, reporting tables and dose figure (manuscript/generated/, manuscript/figures/)
code/number_map.py          every \N{key} in the paper -> value, generated file, source line -> number_map.csv
code/verify_reporting_checks.py    standalone assertions for the reporting-check populations
code/verify_writing_revision.py    standalone assertions for the clustered intervals and one-sided rates
code/informativeness.py     AUROC of the claim vs free features, exposure shortfall -> results/informativeness.csv
code/tolerance.py           verdicts under the three-decimal weights shown in the prompt -> results/tolerance.json
code/paperfig.py            shared figure style for the manuscript figures
code/run_llamacpp.sh        per-model sweep through a dedicated llama.cpp server
code/run_commandr.sh        Command-R-35B default and five-group cells on llama.cpp
code/run_llama70.sh         Llama-3.3-70B default cell on the llama.cpp CUDA endpoint
code/run_qwen3.sh           Qwen3-8B default cell in thinking mode on llama.cpp CUDA
code/run_qwen3_ptrue.sh     Qwen3-8B token-probability run (no per-slate file shipped)
code/run_strong.sh          round-2 stronger models: default, five-group and audit cells
code/scrub_logs.sh          remove local paths and host addresses from run logs before sharing
reproduce.sh                regenerate numbers, rebuild the PDF, diff against the shipped files
check_numbers.sh            regenerate numbers and diff with generated/ (no PDF build)
number_map.csv              key,value,generated_file,script,script_line,used_in for all macros used
generated/                  the five generated .tex files the paper \inputs (numbers, writing numbers,
                            three reporting tables), so the scripts run without the paper source
figures/fig_dose_pgf.tex    dose figure as pgfplots source, generated from the JSON
figures/data/fig_dose_series.json  plotted series behind that figure
revision-data/              recorded outputs of the revision checks: checks.json, pooled_auroc_clustered.json,
                            writing_analysis.json, diagnostic_conditions.csv, projection_utility.csv, source hashes
../manuscript/generated/numbers.tex          every \N{key} macro the paper uses (no digits typed by hand)
../manuscript/generated/writing_numbers.tex  clustered token-probability intervals (kept in their own file)
../manuscript/generated/tab_base_reporting.tex, tab_factors_reporting.tex, tab_audit_reporting.tex
../manuscript/figures/fig_dose_pgf.tex       dose figure as pgfplots source, generated from the JSON
../manuscript/figures/data/fig_dose_series.json  plotted series behind that figure
results/multiplicity.csv    Holm and BH-FDR over the 10 McNemar tests of Table 1
results/multiplicity_table2.csv  Holm over the 45 tests of Table 2
results/frontier.csv        210 frontier points (model x dataset x rho x policy); the `baseline` column marks the three B7 heuristics
results/calibration.csv     B8 Brier score, 10-bin ECE and base rate of P(True) per model x dataset (default cell)
results/split_halves.csv    B9 first/second 200 sampled users: claim, violation, McNemar and token AUROC
results/duplicate_run.csv   B9 Qwen3 base vs ablation_duplicate_client_base on the 54 shared slates
results/feedback_claims.csv B10 base/first/final claim rate and claim AUROC per model and feedback cell
results/slack_strata.csv    violation and claim rates by max-min slack t* (delta = 0.02)
results/position_sensitivity.csv  default-cell violation under four alternative position weights
results/informativeness.csv  claim AUROC, free-feature AUROC and exposure shortfall per cell
results/tolerance.json      verdicts with three-decimal weights against exact weights
results/traces.csv          stated vs true group shares in reasoning traces
results/direction.csv       which way the under-exposed group moves after reranking
results/shortfall.csv       exposure shortfall distribution
results/verifier.csv, results/verifier_shift.csv  synthetic hidden-patience acceptance rules
results/backend_fidelity.json  Ollama vs llama.cpp agreement on shared slates
results/recommender_quality.json, results/recommender_selection.json  recommender validation
results/raw/<model>/<dataset>/<cell>.jsonl   raw model outputs with candidates, constraint, timings
results/ptrue/<model>/<dataset>/<cell>.jsonl token probabilities of the claim
DATA.md                     sources, checksums, licences, how data were collected
PROOFS.md                   proofs of Theorem 1 and Proposition 1
SOLVER.md                   exact-repair ILP, HiGHS version and parameters, problem sizes, timing protocol, hardware
```

## Reproduce

```bash
python3.13 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python code/prep_data.py && ./venv/bin/python code/prep_mind_v2.py && ./venv/bin/python code/prep_history.py
# model servers: llama.cpp `llama-server` (build 9509, 6f3a9f3de) on the Ollama GGUF files of
# qwen2.5:7b, llama3.1:8b, gemma2:9b: Metal at ports 11563/11561/11562, CUDA at 11573/11571/11572.
# Which backend produced each cell is in the `backend` field of every raw record (see DATA.md).
# Weights: Llama-3.3-70B is the bartowski Q4_K_M GGUF; the other six use their default Ollama
# registry tags, 4-bit per the run logs (the paper reports this wording and the source).
code/run_local.sh; for m in qwen2.5:7b llama3.1:8b gemma2:9b; do code/run_queue.sh $m; done
code/run_server_queue.sh llama3.1:8b; code/run_server_queue.sh gemma2:9b; code/run_server_qwen.sh
for m in qwen2.5:7b llama3.1:8b gemma2:9b; do code/run_mind_queue.sh $m; done
for m in qwen2.5:7b llama3.1:8b gemma2:9b; do ./venv/bin/python code/ptrue.py --model $m; done
./venv/bin/python code/analyze.py && ./venv/bin/python code/direction.py && ./venv/bin/python code/fidelity.py
./venv/bin/python code/traces.py
./venv/bin/python code/verifier.py && ./venv/bin/python code/sim_crc.py
./venv/bin/python code/slack_strata.py
./venv/bin/python code/position_sensitivity.py
./venv/bin/python code/frontier.py          # must run before make_assets.py (it writes frontier.csv)
./venv/bin/python code/calibration.py       # must run before make_assets.py (it writes calibration.csv)
./venv/bin/python code/robustness.py
./venv/bin/python code/feedback_claims.py    # must run before make_assets.py (it writes feedback_claims.csv)
./venv/bin/python code/verify_reporting_checks.py
./venv/bin/python code/verify_writing_revision.py
./venv/bin/python code/make_assets.py && (cd ../manuscript && latexmk -pdf main.tex)
```

Seeds are fixed in the scripts (slate sampling, decoding seed, bootstrap, splits). Decoding is
greedy with schema-constrained JSON. Prompts are rendered with Ollama 0.30.8's chat templates for
the three tags, so both backends see the same token sequence (checked by token counts).

### Verify without running a model

The saved outputs are enough to rebuild everything the paper displays. From this directory:

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./check_numbers.sh     # regenerate generated/ and number_map.csv, diff with the shipped files
./reproduce.sh         # same, plus rebuild the PDF and check for ?? and undefined references
```

`check_numbers.sh` rewrites `generated/` only for the duration of the diff and restores it, so a
failure leaves the paper untouched. Both scripts exit non-zero on any difference, on a macro used in
the paper but missing from `generated/`, or on a macro that cannot be traced to a line of
`code/make_assets.py`. They make no model call and no network request.

Neither script needs the paper source: with no `../manuscript`, they compare against the
`generated/`, `figures/fig_dose_pgf.tex` and `figures/data/fig_dose_series.json` copies shipped here,
and print that the macro-usage map and the PDF build were skipped. Placing the paper at
`../manuscript` enables those two checks.

Two standalone checks recompute the headline populations and exit non-zero if one changes:

```bash
./venv/bin/python code/verify_reporting_checks.py   # 10,038/6,552/6,544, Qwen3 14/5/2, audits, feedback
./venv/bin/python code/verify_writing_revision.py   # clustered intervals and one-sided/all-floor counts
```

Both read saved records only (`results/checked.jsonl`, `results/ptrue/*.jsonl`,
`results/feedback_claims.csv`).

`number_map.csv` lists every `\N{key}` in the paper with its value, the generated file that defines
it, the source line in `code/make_assets.py`, and the places in the paper that use it. It is produced
by `code/number_map.py`, which re-runs `make_assets.py` under a line tracer; `script_line` is the
line that first created the macro.

The paper source normally sits next to this directory as `../manuscript` (normally the release
directory is named `artifact/`). `code/make_assets.py` writes to `../manuscript/generated/`; set
`MANUSCRIPT_DIR` to write elsewhere, or edit `MS` at the top of that file if your layout differs.
When the paper is absent, `check_numbers.sh` and `reproduce.sh` regenerate into a temporary copy of
the shipped `generated/` files and skip only the paper-side checks.

## Ablations kept on purpose

- `results/raw/qwen2.5_7b/ml1m/ablation_G*_letterC.jsonl`: an earlier three- and five-group run in
  which one group was named "C" while candidate ids are C01, C02, ...; 40% of the three-group
  outputs were then not valid permutations. Group names now skip "C".
- `results/raw/qwen2.5_7b/mind_tied/`: Qwen on the first MIND version, whose displayed scores were
  tied on most candidates.
- `results/raw/*/mind_nanscores/`: all models on an intermediate MIND version that displayed "nan"
  relevance on 222 of 400 slates. Not analysed.
- `results/raw/llama3.1_8b/ml1m/fidelity_*`, `.../gemma2_9b/ml1m/fidelity_*`,
  `.../qwen2.5_7b/ml1m/fidelity_*`: the same prompts on a second backend (fidelity check).

## Slack and position-weight sensitivity

`code/slack_strata.py` computes a slate-level, objective-free max-min slack
`t* = max_t min_g (E_g(sigma) - floor_g)` with HiGHS. It depends only on the
multiset of candidate groups, `k`, `rho` and the position weights, not on the
relevance scores or the model output. The strata are infeasible (`t* < 0`),
near (`0 <= t* < 0.02`) and wide (`t* >= 0.02`); the shipped
`results/slack_strata.csv` also keeps the old exact-repair margin as an
auxiliary column. On the three small models' exposure cells the infeasible
slates are 276 (all five-group ML-1M slates), with 100% violation and 100%
claims; the near stratum has 473 slates (95.8% violation, 100% claims) and the
wide stratum 16,952 (69.5% violation, 90.0% claims). Qwen3-8B gives one
valid answer to an infeasible slate and reports a violation there, the only
exception.

`code/position_sensitivity.py` re-scores the saved default-cell permutations
under `1/j`, linear `K-j+1`, top-5 and `0.8^(j-1)` weights, keeping the same
`rho` and the same three-decimal floors. This is post-hoc: the models only saw
`1/log2(j+1)`. Pooled over models and datasets, violation moves from 76.4% to
68.7-88.3% on ML-1M and from 83.4% to 78.2-92.8% on MIND, while the claim stays
at 99.7-100%; the qualitative conclusion does not depend on the weight shape.
The paper does not add a sentence for this analysis because the four-page
content limit is already full; the CSV is the artifact record.

## Reporting checks and clustered intervals

`code/reporting_checks.py` recomputes, from `results/checked.jsonl`, `results/ptrue/*.jsonl` and
`results/feedback_claims.csv`, the comparisons the paper reports: the standard-prompt aggregate
under unrounded and displayed weights, the Qwen3-8B default cell, the 276 infeasible five-group
answers, the matched separate audits and the feedback loops. `code/make_assets.py` calls it, so the
three reporting tables and every headline macro in the prose come from the same code path;
`code/verify_reporting_checks.py` asserts the published values and fails if one moves.

The standard-prompt aggregate covers 10,038 valid outputs on feasible slates (6,552 violations with
unrounded weights, 6,544 with the weights displayed in the prompt). Twenty verdicts change when the
prompt's three-decimal weights are used; these repeat 477 slates (292 ML-1M users and 185 MIND
impressions), so the aggregate is not 10,038 independent cases. On the same code path, Qwen3-8B has
14 violations and 2 detections under unrounded weights and 5 violations with 2 detections under
displayed weights, and all 276 valid core-model answers to infeasible five-group slates claim
compliance.

`code/writing_revision.py` computes the stratified token-probability AUROC and a 2,000-replicate
bootstrap (seed 20261115) that resamples slate IDs within each dataset, keeping every variant of a
sampled slate together. It reads only the `ml1m` and `mind` trees under `results/ptrue/`; the same
tree also holds the `mind_tied` and `mind_nanscores` diagnostic runs, which the paper does not pool.
The point estimates are 0.539 [0.509, 0.569] (n = 3,980) for Qwen, 0.430 [0.384, 0.476] (n = 3,981)
for Llama and 0.384 [0.349, 0.420] (n = 3,999) for Gemma. Its `one_sided_rates` helper re-scores the
one-sided-floor outputs against every group floor (103/129, 180/189 and 107/147 violations for the
three models) and supplies the prose macros for that comparison.

The two count-control cells are the input-rate range quoted in the paper: 43.6% of inputs violate
the top-5 count requirement and 60.3% violate the matched-count requirement, which the macros round
to 44--60%. Output violation under those same requirements is 0--4%.

`revision-data/` keeps the recorded output of these checks (`checks.json`,
`pooled_auroc_clustered.json`, `writing_analysis.json`, `diagnostic_conditions.csv`,
`projection_utility.csv`) and the SHA-256 of the records they were computed from. The modules
recompute the same values from `results/`; the directory is the historical record, not an input.

## Baseline heuristics, calibration and robustness (B7-B10)

`code/frontier.py` now also scores three model-independent baselines on the same
feasible slates as the frontier: greedy best-improvement swaps from the input
order until the floor is met, FA*IR through `fairlib.fair_rerank` (the released
`fairsearchcore` package), and equal round-robin over groups with each group in
relevance order. Every row of `results/frontier.csv` has a `baseline` column
(empty for the input/output/projection/exact-repair policies). Utility is the
base-score utility on all feasible slates; nDCG is averaged only over slates
with a held-out positive; `time_ms_median` is wall-clock and is excluded from
the byte-identical check. FA*IR ties for the minority group are broken by group
name so two runs agree. On the shipped slates greedy violates the floor on 0%
of slates at 0.953-1.000 utility, exact repair reaches 0.986-1.000 on the same slate set, FA*IR
violates on up to 85.9% and round-robin on up to 80.5%; the paper states this in
one sentence and does not claim that the heuristics are worse than exact repair
on nDCG.

`code/calibration.py` reads the teacher-forced token probabilities and writes
`results/calibration.csv`: Brier score and 10-bin expected calibration error of
P(True) against whether the output actually satisfies the floor, per model and
dataset in the default cell (pooled and per dataset). Qwen3-8B and Command-R-35B
have no saved token probabilities, so their rows carry `no token probabilities
saved`. The verbal confidence is not calibrated here: it takes only one to three
distinct values, so the paper reports its discrimination by AUROC only.

`code/robustness.py` writes two files. `results/split_halves.csv` splits the 400
sampled users per dataset into the first and second 200 in the frozen slate
order and recomputes the default-cell claim rate, violation, McNemar test and
token AUROC for the three small models. Core cells use the first 300 (ML-1M) or
200 (MIND) users, so the ML-1M second half keeps 92 slates after filtering and
the MIND second half is absent; the CSV records `n=0` rather than imputing it.
`results/duplicate_run.csv` compares Qwen3-8B's base cell with
`ablation_duplicate_client_base`, which DATA.md records as 54 replies from an
accidental second client process on the same slates: 54 slates are shared, 47
are valid in both, violation is 17.0% vs 14.9%, the claim is 100% in both, and
per-slate violation agreement is 97.9%. The conclusion is unchanged in the
halves where it is testable.

`code/feedback_claims.py` reads only the saved `feedback_retry` and
`feedback_memory` records (three small models, ML-1M, 189 slates each) and
writes `results/feedback_claims.csv`. For the base cell and the first and last
round of each loop it reports the violation rate, the rate at which the claim
is `true`, the same rate among violating outputs, and the AUROC of the claim
against compliance. A missing claim counts as `true`, as in `analyze.py`. An
invalid output counts as a violation for the violation rate because it cannot
be certified compliant; claim metrics use valid outputs only. With up to three
retry corrections the final claim is true on 56.4% of Qwen's valid outputs and
37.5% of its violating outputs (claim AUROC 0.80), and on 81.5% and 77.6% for
Gemma (AUROC 0.61); Llama falls to 69.5% on violating outputs but has only 9
compliant slates, below the 10-per-class floor, so its claim AUROC is recorded
as untestable. The memory cell serves the violating output unchanged and leaves
the claim at 98.9-100%. The CSV records the per-cell counts and the reason a
value is missing; the paper uses the same macros and does not infer more than
these 189-slate cells support.

## Projection utility

`code/analyze.py` also solves the same integer program on the model's own order
(scores `k, k-1, ..., 1`), which is the projection reference in the paper. On the
six base cells (three small models x two datasets) it retains 96.8-99.5% of exact
repair's utility; projection minus repair is -1.0 to +1.3 points and the 95%
interval covers zero in every cell. On the ML-1M default slates it changes nDCG
relative to the unprojected order by +1.4 [+0.4, +2.4] for Qwen, +0.9
[-0.5, +2.4] for Llama and +1.1 [+0.1, +2.3] for Gemma. The per-cell values are
the `proj_util`, `proj_ndcg`, `ndcg_minus_proj` and `proj_minus_repair` columns
of `results/summary.csv`; `code/make_assets.py` turns them into the paper macros.

## Release packaging

This directory is the anonymised release copy. Relative to the working copy it has these changes:

- `data/raw/` is excluded because the dataset licences do not allow redistribution. Download the
  archives from the sources in `DATA.md` into `data/raw/` to regenerate the slates and history.
- Item titles are removed from `data/slates_*.json` and from `cands[*]` in `results/raw/`; item ids,
  scores, labels, groups and every result field are kept. The title lists in `data/history_*.json`
  are replaced by `[redacted]` placeholders of the same length.
- In the few cells that ask the model to restate the slate (reasoning-first, user history), the
  model's own output quotes item titles in its `raw`/`work` text. Those titles are replaced by
  `[redacted]` too. The JSON structure and all numbers are preserved, so `code/traces.py`
  regenerates `results/traces.csv` byte-for-byte and `code/make_assets.py` reproduces every
  reported number. A full scan of this directory against all 32,000 dataset titles finds no match.
- `results/logs/`, `.venv/`, `__pycache__/`, `*.log`, `*.time`, scratch scripts and the two
  environment-specific sync/tunnel scripts are omitted. `code/scrub_logs.sh` is kept so that logs
  produced by a new run can be scrubbed before sharing.
- No derived result file was edited. `results/summary.csv`, `results/informativeness.csv` and the
  other derived CSVs are byte-identical to the working copy.

`code/make_assets.py` regenerates `manuscript/generated/numbers.tex`, `writing_numbers.tex` and the
three reporting tables, plus `manuscript/figures/fig_dose_pgf.tex` and its JSON data file, from these
results; `latexmk` then rebuilds the PDF. Nothing displayed in the paper is typed by hand.
