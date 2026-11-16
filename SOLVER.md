# Exact repair and projection: solver, model and timing

This note describes the numeric reference points used in the paper exactly as they
are implemented in `code/fairlib.py` and `code/analyze.py`. It covers what the
integer program decides, how it is built and solved, how a slate that cannot meet
the floor is handled, and how the reported repair time is measured. It is written
from the code and the shipped result files; no model was re-run.

## 1. Where the code lives

| Item | File | Lines |
|---|---|---|
| Floor definition | `code/fairlib.py` | `floors`, `weights`, `shares`, `margin` |
| Exact repair | `code/fairlib.py` | `repair` (60-87) |
| Projection | `code/analyze.py` | `check_record` (60-65) |
| Timing | `code/analyze.py` | `check_record` (38-40) |
| Cell median | `code/analyze.py` | `summarise` (221) |
| Independent max-min slack check (B2) | `code/fairlib.py` | `maxmin_slack` (90-151) |
| FA*IR wrapper (reference point only) | `code/fairlib.py` | `fair_rerank` (154-170) |

## 2. Exact repair (`fairlib.repair`)

### Decision variables

For a slate of `k` candidates, `x[i, j]` is a binary variable that is 1 when
candidate `i` is placed at position `j`. The model is a `k x k` assignment:

* variables: `k^2`, all binary (`integrality = 1`), bounds `[0, 1]`;
* positions are 1-based in the exposure weights, `w_j = 1 / log2(j + 1)`.

### Objective

The solver maximises the weighted base score of the slate,

```
maximise  sum_i sum_j  s_i * w_j * x[i, j]
```

where `s_i` is the base recommender's score for candidate `i` in base-rank order.
This is the DCG of the base scores under the exposure weights. In the code the
minimisation vector is `c = -(outer(scores, w0)).ravel()`.

For **projection**, `analyze.py` calls the same routine as
`repair([k, k-1, ..., 1], g, fl)`, where item `i` is the item at position `i` of
the reranker's output and `g` is the corresponding group sequence. The program
then maximises `sum_i (k - i) * w_j * x[i, j]`, so it keeps the reranker's
preferred items as early as the floor allows. It is not a Kendall
or Spearman distance; it is the same weighted-DCG program with the reranker's
order as the score vector. Projection is only attempted when the base slate is
feasible, which is the same condition, because feasibility depends only on the
multiset of groups.

**Utility evidence.** Across the six base cells (three small models on both
datasets), projection retains **96.8-99.5%** of exact repair's utility; the
projection-minus-repair difference is **-1.0 to +1.3** points and its 95%
interval covers zero in every cell. On the ML-1M default slates it changes nDCG
relative to the model's unprojected order by +1.4 [+0.4, +2.4] for Qwen,
+0.9 [-0.5, +2.4] for Llama and +1.1 [+0.1, +2.3] for Gemma. The columns are
`proj_util`, `proj_ndcg`, `ndcg_minus_proj` and `proj_minus_repair` in
`results/summary.csv`; the paper macros are `proj-retain-min/max`,
`proj-repair-gap-min/max` and the per-cell `-ndcgminusproj` keys.

### Constraints

1. **Each item once:** `sum_j x[i, j] = 1` for every item `i` (`k` equality rows).
2. **Each position once:** `sum_i x[i, j] = 1` for every position `j` (`k`
   equality rows).
3. **Exposure floor:** for every group `g`,
   `sum_{i in g} sum_j w_j * x[i, j] >= f_g * sum_j w_j - 1e-9`.
   The small `1e-9` is the same tolerance used by the verifier (`margin >=
   -1e-9`), so a repaired slate passes the check.

The floors are computed by `floors(groups, rho)` as

```
f_g = floor( rho * n_g / k * 1000 + 1e-9 ) / 1000
```

i.e. `rho * n_g / k` truncated (not rounded) to three decimals, matching the
three-decimal floors shown in the prompt. `rho` is one of 0.5, 0.7, 0.8, 0.9,
0.95.

`repair` also accepts a list of weight vectors (`weight_list`) for a robust
repair against several exposure models. `analyze.py` does not pass it, so every
reported cell uses the single `w_j = 1/log2(j+1)` vector.

## 3. Solver, version and parameters

* Interface: `scipy.optimize.milp` (SciPy `1.18.1`, pinned in
  `requirements.txt`).
* Solver: **HiGHS**, bundled inside SciPy's `_highspy` extension; no separate
  `highspy` package is installed. The bundled shared library reports **HiGHS
  1.12.0** (git hash `4f96ee8`).
* `repair` passes **no `options` dictionary**. SciPy therefore sets only
  `log_to_console = False` and `mip_max_nodes = None`; everything else is the
  HiGHS default:
  * time limit: `inf` (none);
  * node limit: none (`mip_max_nodes` default 2,147,483,647);
  * presolve: HiGHS default `choose` (not fixed to on/off);
  * MIP relative gap: `1e-4`;
  * primal/dual feasibility tolerance: `1e-7`;
  * MIP feasibility tolerance: `1e-6`;
  * parallelism/threads: HiGHS defaults `parallel = choose`, `threads = 0`
    (automatic), so the measurement is **not** a pinned single-thread
    benchmark.

Verified in the development `.venv` on 2026-11-01 (Linux x86_64, Python 3.14):

```text
$ uv pip list --python .venv/bin/python | grep -E 'scipy|fairsearch|highspy'
fairsearchcore     1.0.4
scipy              1.18.1
# no highspy package
$ python -c "import scipy.optimize._highspy._core as C; print(C.HIGHS_VERSION_MAJOR, C.HIGHS_VERSION_MINOR, C.HIGHS_VERSION_PATCH)"
1 12 0
$ strings .../_highspy/_core*.so | grep -E '^4f96ee8$'
4f96ee8
$ python -c "import scipy.optimize._highspy._core as C; o=C.HighsOptions(); print(o.mip_rel_gap, o.primal_feasibility_tolerance, o.mip_feasibility_tolerance, o.presolve, o.parallel, o.threads)"
0.0001 1e-07 1e-06 choose choose 0
```
* The constraint matrix is assembled densely as a `(2k + G) x k^2` NumPy array,
  wrapped in `scipy.optimize.LinearConstraint`, and converted to CSC by SciPy
  before the HiGHS call.
* Repeated solves were not recorded, so run-to-run determinism was not measured.
  The program can have several optimal rankings with the same objective value,
  and `repair` returns `argmax(x[:, j])` for each position without a secondary
  tie-break. A different HiGHS version could therefore return a different
  optimal ranking with the same weighted score; this changes the projected
  order, not the weighted objective or the feasibility verdict.

## 4. Problem sizes

The paper's default cell is `k = 10`, `G = 2`; the variants use `k` in
{5, 10, 20} and `G` in {2, 3, 5}.

| `k` | `G` | Variables `k^2` | Equality rows `2k` | Floor rows `G` | Total rows |
|---|---|---|---|---|---|
| 5 | 2 | 25 | 10 | 2 | 12 |
| 10 | 2 | 100 | 20 | 2 | 22 |
| 10 | 3 | 100 | 20 | 3 | 23 |
| 10 | 5 | 100 | 20 | 5 | 25 |
| 20 | 2 | 400 | 40 | 2 | 42 |

The default cell in the paper is the ML-1M base cell of the three small models:
`k = 10`, `G = 2`, `n = 281` slates each. The repair time printed in the paper
is the **Qwen2.5-7B** row of this cell (see Section 6).

## 5. Slates that cannot meet the floor

`repair` returns `None` when `res.success` is false. `analyze.py` records
`feasible = rep is not None`. In `summarise`, infeasible slates are excluded from
the feasible/valid statistics (`f = d[d.feasible]`) and counted in
`n_infeasible`; valid outputs on infeasible slates are reported separately as
`claim_true_given_infeasible`.

A caveat: `repair` does **not** distinguish "proved infeasible" from a solver
failure; both produce `res.success == False` and `None`. The independent
`maxmin_slack` routine does distinguish HiGHS status 2 (infeasible) and raises on
other failures. The B2 check compared `t* < 0` with the `repair` feasibility flag
on 19,239 valid outputs and found **0 disagreements**, which supports the
classification in the shipped results. The counts are 92 infeasible slates per small model in
the ML-1M five-group cell (276 in total), plus the ablation cells; these are
reported, not imputed.

## 6. How `repair_ms` is measured

`code/analyze.py`, `check_record`:

```python
t0 = time.perf_counter()
rep = repair(bs, bg, fl)
out["repair_ms"] = 1000 * (time.perf_counter() - t0)
```

* Clock: `time.perf_counter()` (monotonic wall clock).
* What is timed: the **entire `repair` call** — building the dense objective and
  constraint matrices, the `LinearConstraint` wrapper, SciPy's conversion to CSC,
  presolve and the HiGHS solve. It excludes JSON parsing, the verifier, `ndcg`,
  the projection call and the bootstrap.
* Iterations: **one call per slate**. There is no warm-up and no repetition
  inside the timed region.
* Per-cell value: `repair_ms_median = d.repair_ms.median()` over **all slates in
  the cell** (including infeasible slates, which still run the solver and return
  `None`). For the **Qwen2.5-7B** ML-1M default cell this is 281 slates; the
  median is **6.682542 ms**, which the paper macro rounds to **7 ms**
  (0 decimals). There is no discrepancy between `summary.csv` and the paper:
  the text uses `\N{qwen-ml1m-base-repairms}` = 7. The paper names
  Qwen2.5-7B in that sentence because the 14B row of the same cell is slower.

Default-cell medians in `summary.csv` (one solver call per slate):

| Model | Median `repair_ms` |
|---|---:|
| Qwen2.5-7B | 6.682542 |
| Llama3.1-8B | 8.803667 |
| Gemma2-9B | 7.380458 |
| Qwen2.5-14B | 10.458625 |
| Command-R-35B | 7.548597 |
| Llama-3.3-70B | 8.457249 |
| Qwen3-8B | 7.782384 |

* The projection solve is a second `repair` call and is **not** included in
  `repair_ms`.
* The timing is a shipped result, not recomputed by `check_numbers.sh` or
  `reproduce.sh`; those scripts re-derive the macros from `summary.csv` but do
  not re-run `analyze.py`.

## 7. Hardware

* Machine: **Apple M5 Max, macOS**. This is author-reported (2026-11-14); no log
  records the host model of the `analyze.py` run. The surviving `time` line is
  in macOS/zsh format (`... 133% cpu 8:43.59 total`), consistent with macOS but
  not evidence of the model. The value was not re-measured for this note.
* The development environment used for the B7-B11 re-analyses of saved outputs
  is Linux x86_64 (not shipped in this release); it is **not** the environment
  that produced `repair_ms`. The macOS run did not log its package versions, so
  `requirements.txt` (SciPy 1.18.1, HiGHS 1.12.0) is the declared/pinned
  environment, not a log of that run. The paper therefore describes the solver
  interface and the version of the shipped environment; it does not claim that
  the timing run logged that version.
* Thread count for the repair is **not recorded**. `repair` does not set a
  thread limit, and the HiGHS default is automatic. The only analysis timing log
  from the working copy (`analyze.time`, not shipped in this release) reports
  133% CPU for a whole `analyze.py --fresh` run, but its `analyze.log` has older
  per-cell counts (Qwen3 base `n_valid = 49` against 98 in the shipped
  `summary.csv`), so it is a different run and is not evidence about the shipped
  measurement.
* Consequence: treat the 7 ms as an order-of-magnitude, single-invocation
  wall-clock median on that machine, not as a single-core benchmark. Do not
  compare it directly with timings from other hardware or other HiGHS versions.

## 8. FA*IR wrapper (reference point, not the repair solver)

`fairlib.fair_rerank` uses the authors' `fairsearchcore` package (`1.0.4`):
protected group = the minority group, `p = n_minority / k` clipped to
`[0.02, 0.98]`, `alpha = 0.1`; documents are sorted by descending score and
passed to `Fair.re_rank`. The released `LegalAssignmentKey.__hash__` returns a
float on Python 3.13+, so the wrapper patches it to return an `int` once. This
path is only used for the `fair_*` reference columns and the B7 baseline; it is
not the exact repair.

## 9. Paper sentences and macro traceability

The deterministic-repair paragraph takes every number from a generated macro:

* Results: `\N{head-repair-ms}` (7 ms median, from `summary.csv`),
  `\N{head-default-attempts-ml1m}` (281 attempts) and `\N{solver-hardware}`
  (Apple M5 Max). They are emitted at `make_assets.py` lines 744, 800 and 52.
* The projection sentence uses `\N{head-repair-n}` (128 slates with held-out
  positives), `\N{head-proj-gain}` (+0.014) and `\N{head-proj-gain-ci}`
  ([0.004, 0.024]), emitted at lines 801-803 from the same `summary.csv` row.
* The hardware string and the solver version are design constants, not measured
  results, but they are still macros so the paper contains no hand-typed
  digits. If a constant changes, update `make_assets.py` and this note together.

`artifact_release/number_map.csv` records the exact source line and the paper
location of every macro key.
