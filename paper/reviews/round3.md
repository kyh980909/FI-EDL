# Review — Round 3 (2026-06-08)

Regression/verification pass over the 11 Round-2 required revisions. Target: `papers/Neurocomputing/FI-EDL_eng.tex` (canonical; `FI-EDL.tex` byte-equivalent; `FI-EDL_kor.tex` parity).

## Recommendation: **Minor revision (borderline Accept)**

All 4 Round-2 correctness/integrity items (SNGP cite, CIFAR-100-ECE abstract, prior-FI-EDL self-reference, stale bib) and the 2 value-add items (compute table, ResNet misclass loss) are fixed and verified. Manuscript is numerically faithful to the results files line-by-line. Two LOW residuals remain (local edits, no re-experiment). **Scientifically/editorially submission-ready** modulo those + local PDF compile + real repo URL at camera-ready.

## Regression check — 11/11 Verified
1. SNGP citation — VERIFIED. Zero orphan `Liu2020` in all 3 .tex (incl. multi-key); 6 `Liu2020sngp` each; bib holds correct Jeremiah Liu NeurIPS 2020; 37 cite keys resolve.
2. CIFAR-100 ECE — VERIFIED. Abstract/§1/§5 now "improving ECE over Re-EDL (−20.74pp, p<0.001), paired-tied on acc + 4 OOD." Matches `cifar100_summary.md`/`cifar100_tin_summary.md`. DAEDL 0.76% ECE framed as 1.76%-acc-collapse artifact (no silent "best ECE").
3. Prior FI-EDL unpublished — VERIFIED. §1/§2.4/§6.1/§8 all "unpublished," no \citep; I-EDL/ℐ-EDL disambiguation footnote restored.
4. Compute-cost table — VERIFIED. `tab:compute` 5 cols, clean rules, every cell traces to `compute_cost.md`; CPU caveat in caption; kor parity.
5. ResNet-18 misclass loss — VERIFIED. §7(ii) "−2.18pp (90.75 vs 92.93, p=0.0003)" matches `resnet_summary.md`.
6. Code/Data availability — VERIFIED. RTX 3060 + framework versions + wall-clock + placeholder URL.
7. CIFAR-10-C p-values dropped — VERIFIED. §5.6 descriptive only; means trace to `cifar10c_summary.md`.
8. α₀ contribution 5 — VERIFIED. "best OOD score depends on the loss family."
9. Highlights 6→5 — VERIFIED. 5 items, all ≤85 chars; kor parity.
10. Nits — VERIFIED. gate caption `λ=β·exp(−γ·tr I)` consistent w/ Eq.(8); figure paragraph break; ASH/GEN/survey cites added and resolve.

## New-error scan
- No dangling \ref/\autoref/\cite; table column counts consistent; eng↔kor parity intact.
- **RESIDUAL 1 [@experimenter][LOW] (pre-existing, not Round-2 introduced):** `tab:main-misc` (eng l.896) lists Re-EDL misclass AUROC = **89.40**, but `stats_significance.md` l.38 uses **89.64**. Both are 5-seed means of the same quantity; reconcile. Does not flip any verdict (no significance claim made against Re-EDL on misclass).
- **RESIDUAL 2 [@writer][LOW]:** §5.1 says F-EDL heads are "~53×" the 5.1K head (270.6K/5.1K≈53, correct), but `compute_cost.md` l.18 Interpretation says "~3×" — the source note is wrong, not the paper. Fix the source artifact so a referee cross-check doesn't trip.

## To reach Accept
1. [@experimenter][LOW] Reconcile Re-EDL misclass AUROC 89.40 vs 89.64.
2. [@writer][LOW] Fix `compute_cost.md` "~3×"→"~53×".
Plus local PDF compile + replace `<anonymized>` repo URL at camera-ready. No new compute required.

## Remaining scientific risks a fresh referee could raise (not mechanics)
- **Scale/backbone breadth:** largest ID benchmark is CIFAR-100; all backbones pre-ViT. Headline "SOTA" = one significant ECE win (ResNet-18, p=0.04) + several paired-ties. A strict referee may read "paired-tied" as "no significant improvement on most metrics" and question whether a *simplification that mostly ties* clears the Q1 novelty bar, given Re-EDL already drops KL and DAEDL/F-EDL already use SN (recipe = recombination of published ingredients; distinct contribution is diagnostic + measurement-convention).
- **Statistics:** n=5 seeds, no multiple-comparison correction across many headline tests; gate-on/masked ablations are n=2.
- **Two losses to DAEDL** (clean misclass AUROC, CIFAR-10-C ECE) both come from the density head the paper argues against → a real counterexample to "simpler is better."
- **DAEDL CIFAR-100 collapse (1.76% acc)** under matched re-implementation vs published 66% will draw re-implementation-fairness scrutiny (partly defused by F-EDL-paper corroboration).
The honest framing pre-empts most; these are predictable second-round pushback, not defects to fix now.

## Confidence: 4/5
All 11 fixes + every reproducible headline number cross-checked against the results files; bib entries verified. −1: no PDF compile executed; 89.40/89.64 unresolved at source.
