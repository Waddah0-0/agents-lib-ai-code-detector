# AI Code Detector

This agent estimates how likely it is that a repository, or parts of it, was written with
AI assistance. It always shows the reasons behind its estimate.

It is in progress: heuristic signals come first (phase 3), and the model-based judge
follows (phase 4).

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
signal scores, so the two stay independent.

The two are then combined into a score for each file and one for the whole repository.

## What it reports

The detector gives a range and a confidence level, such as "0.55 to 0.80, medium
confidence", never a bare percentage. Confidence is low when few signals fired or when
the evidence disagrees.

An explicit marker, like an AI co-author trailer, is reported separately as "AI tool usage
confirmed by metadata". It proves that an AI tool was used, not how much of the code it
wrote.

## Limits

No tool can prove that code was written by AI. Clean human code can look machine-made, and
edited AI code can look human. The detector is built to avoid false accusations, even if
that means it misses some AI-written code.

Use it to inform a conversation. It must not be the sole basis for accusing a student,
a contractor or a job candidate.

## Usage

Not runnable yet. Once phase 3 lands, it will work like this:

```bash
agents run ai-code-detector https://github.com/owner/repo --format markdown
agents run ai-code-detector ./path/to/project --format json --out report.json
```
