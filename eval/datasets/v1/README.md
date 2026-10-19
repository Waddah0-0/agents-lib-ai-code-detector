# Evaluation dataset v1

The phase-4 dataset: used to fit and calibrate the detector's weights (train), to choose
between alternatives (validation), and to report final numbers once per methodology
version (test).

## Composition (70 repositories)

| | Python | JavaScript / TypeScript |
|---|---|---|
| Human, before 2021 | 22 | 10 |
| Human, after 2023 (no-AI policy) | 5 | 0 |
| AI-built, 2025 or later | 16 | 17 |

| Split | Repositories |
|---|---|
| train | 44 |
| val | 13 |
| test | 13 |

Splits are per repository (no repository's files appear in two splits), stratified by
label, language and era, with a fixed seed (`build.py`).

## Sources and labels

- **Human, before 2021** (`sources/human_*_pre2021.jsonl`, plus the 12 from v0): maintained
  open-source projects, each pinned to its last commit before 2021-01-01 with
  `eval/tools/pin_before.py`. Provenance: high (AI coding assistants did not exist yet).
- **Human, after 2023** (`sources/human_python_gentoo_post2023.jsonl`): Gentoo-ecosystem
  Python projects (portage, gentoolkit, mirrorselect, pkgcheck, pkgdev) at their October
  2026 state. Gentoo's council banned AI-assisted contributions on 2024-04-14
  ([policy](https://wiki.gentoo.org/wiki/Project:Council/AI_policy)). Provenance: medium:
  a policy shows intent, not proof, and pkgcheck and pkgdev sit in the separate pkgcore
  organisation. Their code also mixes old and new; the label says the snapshot is post-2023,
  not that every line is.
- **AI-built** (`sources/ai_*.json`, plus the 7 from v0): public repositories created in
  2025 or later where at least 80% of commits carry a verified AI-tool marker, found with
  `eval/tools/find_ai_candidates.py`. Tools represented: Claude / Claude Code, OpenAI Codex,
  the GitHub Copilot coding agent, Devin, Jules and Cursor. Provenance: medium.

## Known gaps and biases

1. **No "mixed" samples** (AI code edited by people). Producing them needs human editors;
   deferred (decision of 2026-10-18).
2. **Era is still mostly confounded with label.** Only 5 human repositories are recent, all
   Python and all from one ecosystem. The era audit (PLAN §9.7) compares them with the
   pre-2021 human set, but on 5 repositories it can only detect large gaps.
3. **AI repositories are selected by their markers.** Markers never enter the estimate, but
   B1 (metadata only) is perfect by construction here, so it measures nothing.
4. **Agent-built projects only.** Code written with light AI help is not represented.
5. **Claude is still the most common tool**, though less dominant than in v0.

## Results (methodology v1)

| | Train (44) | Validation (12) | Test (13) |
|---|---|---|---|
| With the qwen3.5:9b judge: repository AUC | 0.87 | 0.97 | 1.00 |
| False positives at 0.5 | 13% | 29% | 0% |
| Signals only: repository AUC | 0.73 | 1.00 | 0.79 |

Weights were fitted on train, choices made on validation, and the test split scored once
(`eval/results/v1_final_*.json`). `a-my-blog` could not be fetched: its pinned commit was
removed upstream after selection; it is kept in the manifest so the splits do not change.
See `docs/methodology.md` for what these numbers do and do not show.

## Rebuilding

```bash
uv run python agents/ai_code_detector/eval/datasets/v1/build.py
```

## Evaluating

```bash
# development: train and validation splits only
uv run python agents/ai_code_detector/eval/run.py \
    --manifest agents/ai_code_detector/eval/datasets/v1/manifest.jsonl \
    --out agents/ai_code_detector/eval/results/<name>.json

# final report, once per methodology version: add --final to also score the test split
```
