# AI Code Detector

Estimates how likely it is that a repository, or parts of it, was written with AI
assistance, and always shows the reasons behind the estimate.

> **Status: planned.** The heuristic signals and the evaluation harness come first
> (phase 3), then the model-based judge (phase 4). Nothing here runs yet.

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

**A model acting as a judge.** It reviews a sample of files, quotes the specific lines
behind its opinion, and lists the evidence against its own conclusion. It never sees the
signal scores, so the two stay independent. It is only switched on by default if
evaluation shows it makes the results better.

The two are combined into a probability for each file and an estimate for the whole
repository.

## Input

| Field | Default | Meaning |
|---|---|---|
| `target` | required | GitHub URL (`https://github.com/<owner>/<repo>`) or local path. The primary input. |
| `paths` | all files | Restrict the analysis to these files or folders. |
| `branch` | default branch | Branch to analyse (GitHub only). |
| `use_llm_judge` | off | Also ask a model to judge a sample of files. |
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

## Limits

No tool can prove that code was written by AI. Clean human code can look machine-made, and
edited AI code can look human. The detector is tuned to avoid false accusations, even if
that means it misses some AI-written code.

The first version reads Python files only; other languages contribute repository-level
evidence but lower the confidence.

Use it to inform a conversation. It must not be the sole basis for accusing a student,
a contractor or a job candidate.

## Usage

Not runnable yet. Once phase 3 lands, it will work like this:

```bash
agents run ai-code-detector https://github.com/owner/repo --format markdown
agents run ai-code-detector ./path/to/project --format json --out report.json
```

## Development

This agent lives in the [agents-lib](https://github.com/Waddah0-0/agents-lib) monorepo, in
[`agents/ai_code_detector`](https://github.com/Waddah0-0/agents-lib/tree/main/agents/ai_code_detector).
Once it does real work it is also mirrored, read-only, to its own repository; open issues
and pull requests in the main repository.

Run its tests with `uv run pytest agents/ai_code_detector`.
