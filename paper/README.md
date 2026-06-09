# FI-EDL Paper Pipeline — Usage Guide

Four specialized subagents collaborate to take the FI-EDL work to Neurocomputing
submission. Defined in `/home/yongho/FI-EDL/.claude/agents/`.

## The four agents

| Agent | Role | Tools |
|---|---|---|
| `fi-edl-experimenter` | Run new experiments, ablations, statistical tests | Bash, Read/Edit/Write, Grep/Glob |
| `fi-edl-writer` | Draft and revise `main.tex`; integrate experiment results | Read/Edit/Write, WebSearch, WebFetch |
| `fi-edl-reviewer` | Critique drafts as a Neurocomputing reviewer | Read, Grep/Glob, WebSearch, WebFetch |
| `fi-edl-litscout` | Find/summarize related work; maintain `refs.bib` | WebSearch, WebFetch, Read/Edit/Write |

## Invocation

From Claude Code, invoke via the `Agent` tool with `subagent_type`. Examples:

```text
Agent(subagent_type="fi-edl-experimenter",
      prompt="Identify the top 3 experimental gaps for Neurocomputing acceptance and propose a plan with GPU cost estimates.")

Agent(subagent_type="fi-edl-litscout",
      prompt="Populate refs.bib with the EDL lineage and SN/OOD core papers per your scope. Scoop-scan 2024-2026 EDL+SN work.")

Agent(subagent_type="fi-edl-writer",
      prompt="Draft section 4 Method, using results/baseline_comparison_report.md §6.6 for numerical claims.")

Agent(subagent_type="fi-edl-reviewer",
      prompt="Round-1 review of paper/main.tex.")
```

Or in chat: *"experimenter에게 ResNet-18 baseline 추가 검토 부탁"*, *"writer가 Introduction 초안 잡아줘"*.

## Shared state

| File | Purpose | Write access |
|---|---|---|
| `paper/main.tex` | Manuscript prose | `fi-edl-writer` only |
| `paper/refs.bib` | Bibliography | `fi-edl-litscout` only |
| `paper/related_work_notes.md` | Paper summaries + scoop log | `fi-edl-litscout` |
| `paper/TODO.md` | Open items, `[@agent]` tagged | All agents (their own tag) |
| `paper/PAPER_ROADMAP.md` | Milestones M1–M16 | All agents (status updates) |
| `paper/reviews/round{N}.md` | Round-N review | `fi-edl-reviewer` only |
| `paper/reviews/round{N}_response.md` | Author response | `fi-edl-writer` only |
| `results/baseline_comparison_report.md` | **Single source of truth for numbers** | (user / experimenter for §6.6) |

## Recommended workflow

1. **Kickoff (parallel)**:
   - `fi-edl-experimenter` identifies gaps and proposes experiments.
   - `fi-edl-litscout` populates refs.bib core entries and scoop-scans.

2. **Stabilize experiments**: experimenter fills critical gaps (likely ResNet,
   larger-scale dataset, statistical tests). Each experiment ends with a
   `results/<purpose>_summary.md` and a TODO hand-off to writer.

3. **First draft**: writer drafts §1–§8 in `main.tex`, pulling numbers from
   the report and citations from refs.bib. Placeholders `\todo{...}` mark gaps.

4. **First review**: reviewer reads main.tex end-to-end, writes
   `reviews/round1.md` (Summary / Strengths / Weaknesses / Required revisions
   / Questions / Recommendation), and appends per-revision TODO items.

5. **Iterate**: writer addresses revisions; experimenter fills any new gaps;
   litscout fills any new citation gaps. Repeat until reviewer recommends
   "Minor revision" or "Accept".

6. **Submission prep**: writer compiles cover letter, Highlights (≤85 chars
   each), suggested reviewers list, data/code statement, conflict of interest.

7. **(User submits.)** External reviewer reports arrive → writer produces
   `reviews/round{N}_response.md` mapping each comment → response → change.

## Safety rails (built into each agent)

- `fi-edl-experimenter` asks for confirmation if estimated GPU wall-time > 4h.
- `fi-edl-writer` never invents numbers; flags `% TODO @experimenter:` when a
  required number is missing from the report.
- `fi-edl-litscout` never includes a citation without reading the abstract.
- `fi-edl-reviewer` flags claims it cannot verify against the report.

## Honest caveats

These agents draft and analyze; they do not guarantee acceptance. All claims,
numbers, and citations must be verified by you before submission. The agents
treat `results/baseline_comparison_report.md` as ground truth — keep it updated
when new experiments complete.

## Quick start (suggested first commands)

```text
# 1. Have the experimenter identify what's missing
Agent(subagent_type="fi-edl-experimenter",
      prompt="Read MEMORY.md, results/baseline_comparison_report.md §6.6, and paper/TODO.md. Identify the top 3 experimental gaps a Neurocomputing reviewer would flag, with GPU cost estimates for each. Do NOT launch experiments yet.")

# 2. In parallel, have the litscout populate core references
Agent(subagent_type="fi-edl-litscout",
      prompt="Populate refs.bib with the EDL lineage, calibration core, OOD core, and SN core papers per your scope list. Do a scoop scan for 2024-2026 EDL+SN combos. Update paper/related_work_notes.md.")
```
