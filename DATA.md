# Data provenance

All experiments use two public datasets. The raw archives are not included in this release (the
licences do not allow redistribution); download them from the sources below into `data/raw/` to
regenerate the derived slates.

## MovieLens-1M
- Source: https://files.grouplens.org/datasets/movielens/ml-1m.zip (GroupLens), downloaded 2026-10-05.
- SHA-256 of `ml-1m.zip`: `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`
- Size: 1,000,209 ratings, 6,040 users, 3,883 movies (file line counts match the README).
- Licence (README in the archive): research use allowed; acknowledge with the Harper and Konstan
  citation; no redistribution without separate permission; no commercial use without permission.
- Use here: ratings >= 4 are positives; per-user chronological 80/20 split; item-kNN candidate
  generator; provider-group proxy = release period parsed from the title (real metadata, a proxy).

## MIND-small
- Official page: https://msnews.github.io/ . Files used: `MINDsmall_train.zip` from the Hugging Face
  dataset `Recommenders/MIND` (revision `2115068f457b577f93a268fffe22373e45727d89`), the location
  used by the download utilities of the recommenders project (recommenders-team/recommenders,
  `recommenders/datasets/mind.py`, originally Microsoft's), whose dataset card points to the
  Microsoft Research License Terms. Downloaded 2026-10-07 over one connection.
- SHA-256 of `MINDsmall_train.zip`: `6ef97a271580b98ccfc4301ada55cc639423cb0576a78b8dcfcf74a4dbcc3194`
  - `train/behaviors.tsv` `a424547c8fa17c9ea4879c2110221c2b2f2709f4f7615ede0b0d8e4765158658`
  - `train/news.tsv` `cd49d7da275e9c405f627d75a0de9d4ec6ad43ff8a8db5d23a447591676e09d6`
- History of the copy: the experiments up to 2026-10-06 read a copy taken from a third-party mirror
  (`huyva/MIND-small`) because the original Azure blob returned HTTP 409. On 2026-10-07 the train
  files were downloaded from the source above; both TSV files are byte-identical (same SHA-256), so
  no result changes. The mirror copy (including its unused dev split) was deleted.
- Integrity check against the published statistics of MIND-small: train 156,965 impressions and
  51,282 news. Both counts match.
- Licence: Microsoft Research License Terms (`MSR License_Data.pdf` in github.com/msnews/MIND, read
  2026-10-05). Non-commercial research use and publication of results are allowed, provided no
  material portion of the dataset is included in the publication. Distribution of the dataset is
  not allowed.
- Use here: only the train split. Clicks of 9 to 13 Nov 2019, plus the pre-impression click
  history of every user who appears on 14 Nov, train an item-kNN model; slates are real
  impressions of 14 Nov 2019 with their click labels. The displayed relevance blends the kNN score
  with log popularity (weight chosen on 13 Nov, `code/prep_mind_v2.py`). Candidates outside every
  neighbourhood of the user's items get implicit's sentinel score -DBL_MAX; the final version treats
  it as zero similarity. Two earlier versions mishandled it and are kept only as ablations: the
  first min-max scaled the sentinel and collapsed most slates to two score values (`mind_tied`),
  the second propagated -inf and showed "nan" relevance in 222 of 400 slates (`mind_nanscores`;
  not analysed). MIND has no publisher field, so the
  provider-group proxy is the editorial category (news, finance, sports, weather, autos, health
  against all other categories).

## How the data were collected (polite-collection rule of 2026-10-06)
- MovieLens-1M: one HTTPS download of `ml-1m.zip` from files.grouplens.org with curl (default
  curl User-Agent, single connection, no retries needed).
- MIND-small: the original Azure blob answered HTTP 409 "Public access is not permitted on this
  storage account" (2026-10-05); it was not retried. The four TSV files of a third-party Hugging
  Face mirror were then downloaded once each over one connection; that copy was replaced on
  2026-10-07 by one download of `MINDsmall_train.zip` from `Recommenders/MIND` (byte-identical
  train files, see above).
- The MSR licence PDF was read once from github.com/msnews/MIND.
- Literature checks (citation verification only, no data) used the arXiv API and abstract pages,
  Crossref, the ACL Anthology, PMLR, proceedings.iclr.cc, proceedings.neurips.cc and jmlr.org,
  one request at a time. DBLP timed out, ACM DL returned 403 and OpenReview showed a browser check;
  none of these was retried through another route, and Crossref or the proceedings page was used
  instead. One arXiv API request during these checks carried an e-mail address in its User-Agent
  header; later requests did not.
- No other web data was collected. All model outputs were produced locally.

## Release note
In this release, item titles are stripped from `data/slates_*.json` and from `cands[*]` in
`results/raw/` (item ids kept), and the title lists in `data/history_*.json` are replaced by
`[redacted]` placeholders. Item titles that the model echoed into its own `raw`/`work` text in the
reasoning-first and user-history cells are replaced by `[redacted]` as well; the JSON structure and
all numbers are preserved, so `code/traces.py` regenerates `results/traces.csv` byte-for-byte and
every reported number reproduces from this copy. A full scan against all 32,000 dataset titles finds
no match in this directory. `data/raw/` is excluded; see the sources above.

## Synthetic factor
The verifier experiment adds one controlled synthetic factor: a hidden browsing-patience value per
session drawn from {0.5, 0.6, 0.7, 0.8, 0.9} (see `code/verifier.py`). Everything else is real data
and real model output.

## Model backends (one backend per experiment cell; the field `backend` in every raw record)
Weights: the Ollama GGUF files of `qwen2.5:7b` (sha256 2bada8a7...), `llama3.1:8b` (667b0c19...),
`gemma2:9b` (ff1d1fc7...), and for round 2 `qwen2.5:14b` (2049f567...), `qwen3:8b` (a3de86cd...),
`command-r:35b` (8e0609b8...). Files used on the GPU server were downloaded from the Ollama registry
by digest (one connection each) and their SHA-256 matched the digests.

Per cell (all cells of a row use one backend):
- `ollama` (records without a `backend` field): Ollama 0.30.8 on the local machine. Qwen2.5-7B ML-1M:
  tightness sweep, be fair, count, audit, audit at rho 0.8, k 5/20, random/reversed order,
  retry and memory loops; the first MIND version (`mind_tied`).
- `llamacpp`: llama.cpp build 9509 (6f3a9f3de, the binary shipped with Ollama 0.30.8, Metal) on the
  local machine, prompts rendered with Ollama's chat templates. Qwen2.5-7B ML-1M paraphrase, one-sided
  budget, reasoning first, user history, matched count; Llama and Gemma ML-1M tightness sweep and
  be fair; Qwen2.5-14B (prompt rendered by the GGUF chat template, no grammar) on ML-1M.
- `llamacpp_dgx`: a llama.cpp endpoint on the DGX Spark reached on a forwarded local port,
  `Llama-3.3-70B-Instruct-Q4_K_M.gguf`
  (42,520,398,816 bytes, SHA-256 32df3baccb556f9840059b2528b2dee4d3d516b24afdfb9d0c56ff5f63e3a664,
  huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF), llama.cpp commit 6f3a9f3dee3c, CUDA on an
  NVIDIA DGX Spark, flash attention, 4 slots of 4,096 tokens, set up by the project owner; the
  client reached it on a forwarded local port. All Llama-3.3-70B cells (default, five groups, audit) and their token
  probabilities. Prompts rendered by the GGUF chat template (leading BOS text stripped because
  /completion adds BOS), no grammar.
- `llamacpp_cuda`: the same llama.cpp commit built with CUDA on a rented H100 MIG slice. All MIND
  cells of all three small models; all other Llama and Gemma ML-1M cells (variants, audit, count,
  k, groups, order, loops); Qwen2.5-7B ML-1M G3 and G5; Qwen3-8B (thinking mode, client on the
  server). No per-slate Qwen3-8B token probabilities are shipped, so its AUROC rows are empty
  rather than imputed; two artefacts of the Qwen3 runs are kept: `ablation_budget8k_base` (an 8k budget, abandoned
  because 8 of 13 replies ran out of budget) and `ablation_duplicate_client_base` (54 replies from
  an accidental second client process; the first reply per slate is used).

Fidelity (results/backend_fidelity.json, identical prompts on two backends, 572 slates in eight
pairs covering Ollama vs local llama.cpp, local llama.cpp vs CUDA, and Ollama vs CUDA): the
violation verdict agrees on 553 (96.7%) and the exact ranking on 431 (75%), because greedy
decoding diverges after small numerical differences. (The claim agrees on all 572, which says
little because the claim is nearly constant.) Token probabilities of the claim are read by teacher
forcing on the local llama.cpp backend: the prompt plus the model's own raw output up to the closing
colon of `"satisfied":` is scored with a one-token continuation, the top-20 log-probabilities are
summed with `exp` over the tokens `true` and `false`, and the reported value is
`p_true / (p_true + p_false)`, i.e. the probability of the claim conditional on the field taking one
of the two boolean values. Probabilities are shipped for Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B,
Qwen2.5-14B and Llama-3.3-70B; Qwen3-8B and Command-R-35B have none, so their AUROC rows are empty
rather than imputed. `results/ptrue/<model>/` also keeps the two MIND ablations (`mind_tied`,
`mind_nanscores`); `code/writing_revision.py` pools only the `ml1m` and `mind` directories, so the
clustered interval in the paper excludes the diagnostic variants.

## Split-half and duplicate-run provenance
The slate files keep the sampled users in their frozen order. `code/robustness.py` splits that
order into the first and second 200 users per dataset and reuses only the saved default-cell
outputs; it runs no model. Core cells use the first 300 ML-1M users and the first 200 MIND users, so
the ML-1M second half has 92 slates left after filtering and the MIND second half was never run; the
CSV records `n=0` for it rather than imputing a result. The same script reads
`ablation_duplicate_client_base`, which this file records as 54 replies from an accidental second
client process on the same Qwen3-8B ML-1M slates (the first reply per slate is used in the base
cell). Of the 54 shared slates 47 are valid in both; the script reports violation, claim rate and
per-slate agreement there. Neither check changes any number in the manuscript.

## Feedback-loop claim provenance
The retry and memory cells are saved under `results/raw/<model>/ml1m/feedback_retry.jsonl`
and `feedback_memory.jsonl`. Every record keeps a `rounds` list, and every round stores the
parsed `ranking`, `valid`, `claim`, `confidence` and `violated` fields, so the first and last
rounds can be compared without rerunning a model. The retry cell asks again after a violation,
up to three times; the memory cell is FACTER-style and serves the violating output unchanged,
so it has one round. `code/feedback_claims.py` uses those saved fields only. It treats a missing
claim as `true` (the convention in `analyze.py`), counts an invalid output as a violation for the
violation rate, and excludes invalid outputs from the claim metrics. Llama's retry final round
has 9 compliant slates, below the 10-per-class AUROC floor, so its claim AUROC is left missing
rather than estimated. These cells are ML-1M only and cover 189 slates per model.
