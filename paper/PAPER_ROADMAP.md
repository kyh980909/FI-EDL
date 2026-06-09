# FI-EDL Neurocomputing Journal Extension — Roadmap

**Target venue**: Neurocomputing (Elsevier, Q1 CS Applications, IF ≈ 5–6)
**Status**: M1 complete — pipeline scaffolded 2026-05-29
**Submission target**: TBD (typical 8–12 weeks from kickoff to first submission)

## Verified recipe (from results/baseline_comparison_report.md §6.6)

| Dataset | backbone_spectral_norm | loss | OOD score | Result vs SOTA |
|---|---|---|---|---|
| CIFAR-10 | **true** | fi_edl (gate-off default) | alpha0 | **#1 ECE / #1 SVHN / #1 C100 / #2 Misclass / top-tier Acc** |
| MNIST | false | fi_edl | alpha0 or maxp | #2 ECE; OOD 6–7th (DAEDL dominant via density) |

## Milestones

| # | Milestone | Owner | Status | Notes |
|---|---|---|---|---|
| M1 | Pipeline scaffold (agents, paper/, ROADMAP, TODO, refs.bib stub, main.tex skeleton) | (Claude main) | ✅ DONE 2026-05-29 | |
| M2 | Experimental-gap identification | @experimenter | TODO | Likely: ResNet on CIFAR/CIFAR-100/TinyImageNet, statistical tests, extra OOD sets |
| M3 | Initial literature scan + refs.bib core entries | @litscout | TODO | EDL lineage + SN + OOD + calibration core, plus scoop scan |
| M4 | Fill critical experimental gaps | @experimenter | TODO | After M2 priorities confirmed |
| M5 | Section outline + Abstract draft | @writer | TODO | One paragraph per section + numbered contribution list |
| M6 | §3 Background + §4 Method draft | @writer | TODO | |
| M7 | §5 Experiments + §6 Ablations draft (uses M4 outputs) | @writer | TODO | |
| M8 | §1 Introduction + §2 Related Work + §7–§8 draft (uses M3) | @writer | TODO | |
| M9 | Highlights + Abstract finalization | @writer | TODO | Elsevier required 3–5 bullets, ≤85 chars each |
| M10 | Round-1 internal review | @reviewer | TODO | Output: paper/reviews/round1.md |
| M11 | Address Round-1 revisions | @writer + @experimenter | TODO | |
| M12 | Round-2 review | @reviewer | TODO | Should reach Minor revision recommendation |
| M13 | Final polish + cover letter + suggested reviewers + Highlights + data/code statement | @writer | TODO | |
| M14 | Submission to Neurocomputing | (user) | TODO | |
| M15 | Reviewer reports → response document → revisions | @writer + @experimenter | TODO | Typically 1–2 rounds |
| M16 | Acceptance | (Neurocomputing) | TODO | |

## Standing requirements (Neurocomputing)
- **Highlights**: 3–5 bullets, each ≤85 characters, capturing key findings.
- **Cover letter**: explain novelty, fit with journal scope, and (if applicable) the prior conference paper this extends.
- **Suggested reviewers**: 3–5 names with affiliations and brief justification. Avoid direct conflicts.
- **Conflict of interest** statement.
- **Data/code availability** statement — release code on GitHub (placeholder URL OK at submission, must resolve by acceptance).
- **Graphical abstract** (optional but recommended).

## Risk register
- **Scoop risk**: a 2025–2026 paper combining EDL + SN may exist; @litscout to scan.
- **MNIST regression**: SN hurts MNIST ECE. Must be honestly discussed (§6.3 ablation + §7 limitation). Reviewers will notice; pre-empting is better than being asked.
- **Single-backbone risk**: only VGG-16 (CIFAR) + ConvNet (MNIST). Reviewers will ask for ResNet — @experimenter must address.
- **Statistical significance**: paired tests not yet in the report. @experimenter to add.

## Suggested reviewers (working list — @writer to refine before submission)
- Authors of R-EDL and Re-EDL (mid-overlap; can assess novelty fairly)
- Authors of SNGP / DUQ (SN+uncertainty expertise)
- Avoid: DAEDL group (direct competitor), F-EDL group (direct competitor) — too close
- Avoid: prior FI-EDL paper reviewers if known
