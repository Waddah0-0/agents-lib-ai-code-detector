# AI Code Detector

Estimates how likely it is that a repository, or parts of it, was written with AI
assistance, and always shows the reasons behind the estimate.

> **Status: experimental.** Methodology v1 combines heuristic signals with a local LLM judge,
> with weights fitted on 44 repositories. On 13 held-out repositories it ranked every AI
> repository it scored above 0.5 higher than every human one (AUC 1.00, no false positives
> at 0.5), but that test set is small and the scores are not calibrated, so confidence never
> goes above medium. Without a model it falls back to much weaker signals (test AUC 0.79,
> 29% false positives). Read [how it scores](https://github.com/Waddah0-0/agents-lib/blob/main/docs/methodology.md)
> before relying on any number.

## What it does

Given a GitHub repository URL or a local path, the detector looks for evidence of AI
assistance in the code and its history. It returns a range with a confidence level, plus
the evidence for and against, so a person can judge the result for themselves.

## How it works

The detector combines two kinds of evidence.

**Cheap, explicit signals.** These look at the repository itself. Examples:

- config files left behind by AI coding tools, such as `CLAUDE.md`, `.cursorrules` or
  `.github/copilot-instructions.md`;
- `Co-authored-by` trailers that name an AI assistant;
- a whole project arriving in a single commit;
- comments that only restate the line below them.

Some signals point the other way: a long, messy history, personal TODOs and leftover debug
code are all evidence of human authorship, and they count too.

**A model acting as a judge.** A local model (default `qwen3.5:9b` via Ollama) reviews up to
20 files, points at the lines behind its opinion, and lists evidence against its own
conclusion. The quotes in the report come from the file, so evidence cannot be invented.
It never sees the signal scores. It runs automatically when a model is available, because in
evaluation it was the only evidence that worked on recent code.

The two are combined into one estimate for the repository, with weights fitted on data.
Commit history and README style are shown as context but not scored: in evaluation they
told old projects from new ones, not people from AI.

## Input

| Field | Default | Meaning |
|---|---|---|
| `target` | required | GitHub URL (`https://github.com/<owner>/<repo>`) or local path. The primary input. |
| `paths` | all files | Restrict the analysis to these files or folders. |
| `branch` | default branch | Branch to analyse (GitHub only). |
| `use_llm_judge` | when a model is available | Ask a model to judge a sample of files. `false` skips it (weak, low confidence); `true` requires it. |
| `include_tests` | off | Score test files too (they are reported separately). |

The full list, including the judge's file, time and cost limits, will be in the agent's
input schema (`agents info ai-code-detector`).

## Output

- **An estimate with a range and a confidence level**, such as "0.55 to 0.80, medium
  confidence", never a bare percentage. Confidence is low when few signals fired, few
  files could be analysed, or the evidence disagrees, and the report says which.
- **"AI tool usage confirmed by metadata"**, shown separately when an explicit marker such
  as an AI co-author trailer is found. It proves a tool was used, not how much code it
  wrote, so it does not change the estimate.
- **Evidence for and evidence against**, each pointing to a file, line range or commit.
- **A score per file**, with the signals that fired and, when the judge ran, its quotes.
- **Coverage**: how many files were analysed or skipped, by language.
- **The code behind the evidence**: short excerpts around each marked line (capped at 80
  lines per file and 800 in total), so a report can be checked without the repository.
- **A commit list for a timeline**: date, subject, files changed, AI-tool markers, and whether
  the commit was unusually large. Author names are left out on purpose: the report is about
  code, not people.
- **Every unscored file and why** (unsupported language, test file, too small, lockfile...).

## Limits

No tool can prove that code was written by AI. Clean human code can look machine-made, and
edited AI code can look human. The detector is tuned to avoid false accusations, even if
that means it misses some AI-written code.

It reads Python, JavaScript and TypeScript; other languages lower the confidence. Projects
with large generated-looking data files (such as locale tables) can score high. It has not
been tested on code written with light AI help or on AI code edited by people.

Use it to inform a conversation. It must not be the sole basis for accusing a student,
a contractor or a job candidate.

## Usage

```bash
agents run ai-code-detector https://github.com/owner/repo --format markdown
agents run ai-code-detector ./path/to/project --format json --out report.json
```

## Development

This agent lives in the [agents-lib](https://github.com/Waddah0-0/agents-lib) monorepo, in
[`agents/ai_code_detector`](https://github.com/Waddah0-0/agents-lib/tree/main/agents/ai_code_detector).
It is mirrored, read-only, to
[Waddah0-0/agents-lib-ai-code-detector](https://github.com/Waddah0-0/agents-lib-ai-code-detector);
open issues and pull requests in the main repository.

The evaluation harness lives in `eval/`: `eval/run.py` scores a dataset manifest,
`eval/fit.py` fits and compares methodologies, `eval/tools/compare_judges.py` compares judge
models, and `eval/datasets/*/README.md` describe each dataset and its biases.

Run its tests with `uv run pytest agents/ai_code_detector`.
