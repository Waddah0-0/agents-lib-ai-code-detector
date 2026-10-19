# Evaluation dataset v0

A small first dataset for **baseline measurements only** (PLAN §9.7). It is too small and too
biased to fit or calibrate weights; that is the job of dataset v1 in phase 4.

## Composition

19 repositories: **12 human** and **7 AI**. The plan asked for 10 AI samples; the search
found only 7 that passed every criterion before GitHub's unauthenticated rate limit made
further searching impractical. Five of the seven were built with Claude (one also with
Codex) and two with Devin, so the AI side leans heavily on one tool family.

- **Human:** established Python projects, each pinned to its last commit before
  2021-01-01 (before AI coding assistants were available). Label `human`, era `pre2021`.
- **AI:** public Python repositories created in 2025 or later in which at least 80% of
  commits carry a verified AI-tool marker (co-author trailer, "generated with" line or bot
  author; see `markers/v1.toml`). Label `ai`, era `post2023`.

Every sample is a GitHub URL plus an exact commit; no third-party code is stored here. The
fetcher clones each one (blobless) into the cache directory (`MARGIN_CACHE_DIR`,
default `~/.cache/margin/eval`).

## How the samples were chosen

- Human: `eval/tools/pin_before.py --before 2021-01-01T00:00:00Z <repos>`, over a hand-picked
  list of maintained, mostly mid-sized Python projects.
- AI: `eval/tools/find_ai_candidates.py`, which searches GitHub commits for each verified
  marker and keeps repositories that pass the criteria in its docstring.

Selecting AI samples by their markers does not leak the label into the score: the
detector's estimate never uses markers. It does mean every AI sample *also* has markers,
so baseline B1 (metadata only) is perfect here by construction. B1 measures nothing on v0.

## Known biases (read before quoting any number)

1. **Era confound.** Every human sample is from 2020 or earlier and every AI sample is from
   2025 or later. A detector could separate them by era (newer idioms, type hints, newer
   libraries) rather than by authorship. v1 adds matched pairs and post-2023 human code to
   measure this.
2. **AI samples are agent-built projects with agent-made histories.** Code that a person
   wrote with light AI help, or AI code that was heavily edited, is not represented.
3. **Popular human projects.** Well-known libraries are carefully documented, which looks
   "AI-like" to comment-based signals, so v0 may overstate false positives on ordinary
   human code. Their code is also likely in LLM training data, which matters for the judge
   in phase 4.
4. **Tiny and unbalanced.** 19 repositories (12 human, 7 AI); two AI repositories
   contribute most of the AI files. Differences of a few points in AUC are noise.
5. **Python only**, and labels are per repository (files inherit them).
6. **One tool family.** Most AI samples were built with Claude.

## Baseline results (methodology v0)

Full results: `eval/results/v0.json`.

| | Repository AUC | False positives at 0.5 | At 0.6 (FPR / TPR) | Calibration error |
|---|---|---|---|---|
| B0 constant prior | 0.50 | | | |
| B1 metadata only | 1.00 (by construction, see above) | | | |
| B2 heuristic signals | 0.86 | 4 of 12 human repos (33%) | 8% / 29% | 0.17 |

File level (B2): AUC 0.64 over 1,106 files. Pre-2021 human repositories average 0.47 and
post-2023 AI repositories 0.57, a gap that may partly reflect era rather than authorship.
The weights were not tuned on this dataset, so these numbers are honest baselines.

## Running it

```bash
uv run python agents/ai_code_detector/eval/run.py \
    --manifest agents/ai_code_detector/eval/datasets/v0/manifest.jsonl \
    --out agents/ai_code_detector/eval/results/v0.json
```
