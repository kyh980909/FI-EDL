# Open Items — FI-EDL Paper

Format: `- [ ] [@agent] task — context/ref`. Check off when done. Add date when added.

## High priority — round 1 revisions

> Source: `paper/reviews/round1.md` (2026-05-29). Recommendation: Major revision.
> Each item below is a numbered `Required revision` from the round-1 review.

- [x] (2026-05-29 main agent) R1 — Reconciled gate magnitude across Abstract / §1 / §4.1: now reports MNIST $\overline\lambda\!\approx\!4{\times}10^{-7}$ (3 orders below fit) and CIFAR-10 $\overline\lambda\!\approx\!4{\times}10^{-4}$ (~70% of fit) separately; reframed "controller off" as "controller acts as a mis-tuned constant scalar; no per-sample adaptivity (trace std 1.6% MNIST / 6.5% CIFAR-10)".
- [x] (2026-05-29 main agent) R2 — Removed "(below the EDL baseline)" in §1 and §4.3; corrected 80.6 → 83.80 for CIFAR-100 baseline FI-EDL maxp. Numbers re-verified against `results/table2_extended.csv` (FI-EDL maxp SVHN=89.89, FI-EDL maxp C100=83.80).
- [x] (2026-05-29 main agent) R3 — Wrote `results/score_audit.md`. Per-method best score per OOD set: EDL/I-EDL/R-EDL/F-EDL → maxp; Re-EDL/DAEDL/FI-EDL → alpha0. Awaits writer to integrate into §5.3.
- [x] (2026-05-29 main agent) R4 — Added mechanism paragraph at end of §4.3 ("Why does the minimal recipe beat a more elaborate baseline?"): two hypotheses, $(p,\tau)$ gradient interference and Flexible-Dirichlet evidence-splitting weakening $\alpha_0$ semantics. Labelled "we propose these as hypotheses".
- [x] (2026-05-29 main agent) R5 — Abstract last paragraph now acknowledges $-1.29$ pp Misclass loss to DAEDL explicitly, with the carved-out title scope ("calibration and OOD detection, with one acknowledged loss on misclassification").
- [x] (2026-05-29 main agent) R6 — Added Code and Data Availability paragraph at end of §5.1 with anonymized URL placeholder, dataset list, per-seed wall-clock times, and framework version note.
- [x] (2026-05-29 main agent) R7 — Scoped Abstract and §1 contribution 3 to "CIFAR-10/VGG-16"; added "Generalisation to ResNet-18 is reported in the same section" forward-reference for when Gap 1 lands.
- [x] (2026-05-29 main agent) R8 — Highlight 3 now reads "Best-mean CIFAR-10 ECE 3.93\% (paired-tied with DAEDL), SVHN AUROC 92.99 ($p{=}0.09$)" — qualified within 85 chars. Highlight 1 also updated to focus on the "fails to adapt" story (trace std 6.5\% of mean) rather than the previous $5{\times}10^{-7}$ point estimate.
- [x] (2026-05-29 main agent) R9 — Replaced both `results/stats\_significance.md` inline references with "the supplementary material". Supplement file to be created at submission time from `results/stats_significance.md` and `results/score_audit.md`.
- [x] (2026-05-29 litscout round 2) R10 — Added three verified 2023+ entries to refs.bib: `liu2023cals` (Class Adaptive Network Calibration, CVPR 2023; train-time margin-based calibration), `djurisic2023ash` (Activation Shaping, ICLR 2023; post-hoc feature-shaping OOD), `liu2023gen` (Generalised Entropy score, CVPR 2023; post-hoc softmax-based OOD). Suggested §2.3 LaTeX integration snippet handed to writer (see litscout output). Abstracts verified via WebFetch.
- [x] (2026-05-29 litscout round 2) R11 — `evisec2025` placeholder resolved: real author list extracted from arXiv 2506.07417 (Sun, Lin, Zhou, Shang, Cheng, Cao) and ECML-PKDD 2025 acceptance confirmed; bibkey renamed to `sun2025evisec`. `chen2024reedl` upgraded to journal: TPAMI 2025 acceptance verified via authors' official GitHub repo and IEEE Xplore document 11052867; bibkey renamed to `chen2025reedl` and main.tex sed-updated. No "preprint" caveat needed.
- [x] (2026-05-29 main agent) R12 — Verified §1 89.89 (FI-EDL maxp SVHN) and 83.80 (FI-EDL maxp C100) against `results/table2_extended.csv`. Both match. The previous "80.6" was a writer typo for 83.80; corrected in R2.
- [x] (2026-06-08 verified) R13 — `tab:abl-gate-on` and `tab:abl-masked` already carry $^{\dagger}$ markers on every $n\!=\!2$ row + the "$n\!=\!2$ seeds; baseline row uses $n\!=\!5$" footnote, in BOTH FI-EDL_eng.tex and FI-EDL_kor.tex (9 dagger occurrences each, parity confirmed). TODO was stale; content was done in the Option-B rewrite.
- [x] (2026-05-29 main agent) R14 — Algorithm `alg:recipe` precondition moved out of training pass into a top-line `\KwSty{Setting:}` block with explicit "omit SN for small backbones" instruction.

## Round 2 revisions (2026-06-08, main agent)

> Source: `paper/reviews/round2.md`. Applied to all three Neurocomputing .tex
> (`FI-EDL_eng.tex`, byte-identical `FI-EDL.tex`, Korean `FI-EDL_kor.tex`).

- [x] R2-1 [HIGH] Repointed all 6 `\citep{Liu2020}`→`\citep{Liu2020sngp}` (SNGP). 0 `Liu2020}` remain in all three .tex.
- [x] R2-2 [HIGH] Fixed CIFAR-100 ECE "tie" misstatement in Abstract + §1 contrib 3: now "paired-tied with Re-EDL on accuracy + four OOD, improving ECE over Re-EDL (−20.74pp, p<0.001)"; §5 body adds degenerate-DAEDL caveat (DAEDL 0.76% ECE = artifact of 1.76% acc collapse). Numbers from `cifar100_summary.md`.
- [x] R2-3 [HIGH] Prior FI-EDL stated as "earlier, unpublished formulation by the present authors" at §1/§2.4/§8 first mentions; added I-EDL (Deng2023) disambiguation footnote + §2.4 sentence. No self-\citep added.
- [x] R2-4 [MED] Added compute-cost table `tab:compute` to §5.1 (FI-EDL+SN 5.1K head vs F-EDL 270.6K; CPU-timing caveat in caption). From `compute_cost.md`.
- [x] R2-5 [MED] Added ResNet-18 misclass loss −2.18pp (90.75 vs 92.93, p=0.0003) to §7 limitation (ii). From `resnet_summary.md`.
- [x] R2-6 [MED] Strengthened §5.1 Code/Data availability: RTX 3060, PyTorch 2.11/CUDA 12.8, Lightning 2.6, sklearn 1.8, SciPy 1.17 (from uv.lock), wall-clock (from CODE_RELEASE_README), anonymized repo URL.
- [x] R2-7 [MED] Dropped all CIFAR-10-C p-values in §5.6; reworded descriptive (mean diffs only). Means verified against `cifar10c_summary.md` (16.81/14.43/19.72/19.52).
- [x] R2-8 [LOW] §1 contrib 5: α₀ reworded to "best OOD score depends on the loss family" (from `score_audit.md`).
- [x] R2-9 [LOW] Highlights merged 6→5 (≤85 chars each).
- [x] R2-10 [LOW] (a) `fig:gate-diagnosis` caption now matches Eq.(8) β,γ. (b) blank line inserted after `\end{figure}` at the reliability-panel prose. (c) Integrated `\citep{Djurisic2023}`(ASH)+`\citep{Liu2023gen}`(GEN) into §2.3, `\citep{Gao2024survey}` into §2.1.

## High priority — kickoff

- [x] [@experimenter] (2026-05-29) Gap analysis — output in results/gap_analysis.md.
- [x] [@experimenter] (2026-05-29) **Gap 2 DONE** — paired t-test + bootstrap CI; results/stats_significance.md written, report §6.6.7 added. Key findings: only CIFAR-100 OOD is statistically uniquely #1; ECE/SVHN are best-mean but tied with top competitor; Misclass is significantly #2 (DAEDL lead 1.29pp). Writer must report honestly.
- [ ] [@experimenter] (2026-05-29) **Gap 1 RUNNING** (~22h GPU) — ResNet-18 sweep launched 2026-05-29; FI-EDL+SN seed 0 at epoch 13, val/acc 0.87. Aggregation script ready at scripts/aggregate_resnet_gap1.py.
- [ ] [@experimenter] (2026-05-29) **Gap 3 READY** — DTD OOD loader added to src/data/adapters/cifar10.py (also Places365); driver scripts/eval_gap3_dtd.sh ready. Forward-only eval, awaits Gap 1 completion to avoid GPU contention.
- [x] [@litscout] (2026-05-29) Populate refs.bib with EDL lineage (Sensoy18, Malinin18, Amini20, I-EDL=Deng23 ICML [venue corrected from 2021], R-EDL Chen24, Re-EDL Chen24, DAEDL Yoon24, F-EDL Yoon25), calibration (Guo17, Mukhoti20), OOD (Hendrycks17, Liang18 ODIN, Lee18 Mahalanobis, Liu20 SNGP, Sun22 KNN), SN+Lipschitz (Miyato18, vanAmersfoort20 DUQ, Bartlett17). 17 verified entries + 2 adjacent 2025 arXiv items added.
- [x] [@litscout] (2026-05-29) Scoop scan complete. **TWO direct overlaps:** DAEDL (Yoon24 ICML) applies SN to EDL feature extractor. F-EDL (Yoon25 NeurIPS) applies SN + replaces KL with Brier-on-mean + input-dependent priors — 3 of our 4 ingredients overlap. Flagged with `% RISK: potential scoop` in refs.bib; differentiation strategy in related_work_notes.md. **NAMING ALERT for @writer:** "F-EDL" abbreviation now belongs to Yoon&Kim 2025 NeurIPS (Flexible EDL). Our paper must consistently use "FI-EDL" (Fisher-Information EDL) and never "F-EDL".
- [x] [@writer] (2026-05-29) **Round 2 done** — Drafted Abstract (~290 words), 5 Highlights (≤85 chars each), §1 Introduction with the 5-item numbered contributions list and paper outline paragraph (~1.5 pages), §2 Related Work in 3 subsections including the head-to-head `tab:rw-concurrent` table that defuses the DAEDL/F-EDL scoop, and §6 Ablations with three tables (`tab:abl-gate-on`, `tab:abl-masked`, `tab:abl-sn`). Added `pifont`, `amssymb`, `multirow` to preamble for the new tables. Followed the litscout naming convention: FI-EDL for Fisher-information EDL, F-EDL for Flexible EDL (Yoon 2025 NeurIPS).
- [x] [@writer] (2026-05-29) **Drafted §3 Background, §4.1 Diagnosis, §4.2 Recipe, §4.3 Why each component, §5.1 Setup** — replaces 5 \todo markers in main.tex with full LaTeX prose (~3 pages). Adds `algorithm2e` to preamble. Leaves `\todo{@experimenter}` for ResNet-18 results in §5.1.

## Medium priority

- [ ] [@experimenter] CIFAR-10-C robustness ECE benchmark (15 corruptions × 5 severities) if compute permits.
- [ ] [@experimenter] Compute cost comparison: training time and inference time (FI-EDL+SN vs DAEDL+GMM vs F-EDL).
- [x] [@writer] (2026-05-29) Highlights — 5 bullets, ≤85 chars each (round 2).
- [x] [@writer] (2026-05-29) Draft §3 Background (Dirichlet/EDL formulation; KL-to-uniform; FIM trace formula with a 1-paragraph derivation).
- [x] [@writer] (2026-05-29) Draft §4 Method (4.1 diagnosis, 4.2 recipe, 4.3 justification).
- [ ] [@writer] Draft §5 Experiments — §5.1 Setup done; **§5.2 Main results (Tables 1--3) and §5.3 Score selection ablation still pending** — await final ResNet-18 numbers from Gap 1 + score-ablation aggregate from @experimenter.
- [x] [@writer] (2026-05-29) Draft §6 Ablations (gate-on collapse, masked-KL, SN on/off) — three tables rendered from report §6.6.3/§6.6.4/§6.6.5.
- [ ] [@writer] Draft §7 Discussion — when SN helps (relate to SNGP/DUQ Lipschitz argument), limitations (MNIST regression, single architecture family per dataset, no ImageNet), future work (learned SN coefficient, hybrid with density à la DAEDL, ResNet/ImageNet scaling).
- [ ] [@writer] Draft §8 Conclusion — restate the three-component recipe; one sentence on the C100 statistical SOTA + paired-tie framing on ECE/SVHN; one sentence on the honest MNIST SN regression.
- [ ] [@writer] After §5.2/§5.3/§7/§8 are drafted, sweep main.tex for stale `\autoref` targets and missing cross-references.
- [ ] [@litscout] After main draft: scan for recent citations in §2 Related Work that might have changed.

## Backlog

- [ ] [@experimenter] If time permits: ImageNet-100 (subset) to demonstrate scaling.
- [ ] [@writer] Cover letter draft.
- [ ] [@writer] Suggested reviewers list (3–5 names with affiliations + reason).
- [ ] [@writer] Data/code availability statement; create code repo URL placeholder.
- [ ] [@writer] Graphical abstract (optional).
- [ ] (user) Confirm authorship order and corresponding author affiliation.
- [ ] (user) Decide whether to cite the conference FI-EDL paper as an extension (Neurocomputing allows this; must disclose).

## Brier-score column (2026-06-09, writer)

- [x] Added `Brier $\times 100\,\downarrow$` column (after NLL) to the CIFAR-10/VGG-16 and MNIST Table-1 calibration tables, and (after ECE) to the CIFAR-100/ResNet-18 table, in all three .tex (`_eng`, byte-identical `FI-EDL.tex`, Korean `_kor`). Numbers traced to `results/brier_summary.md`. Bolded best (lowest) per column: FI-EDL on CIFAR-10 (15.36) and CIFAR-100 (45.78); F-EDL on MNIST (1.02) — FI-EDL NOT bolded on MNIST. Added prose: calibration-metric caption updates, objective-alignment caveat, F-EDL CIFAR-10 paired-tie (p=0.63), MNIST non-best note, DAEDL-C100 Brier-degeneracy note. Applied `\setlength{\tabcolsep}{4pt}` to MNIST + CIFAR-100 tables to avoid overflow; CIFAR-10 table* fit at default. Abstract/Highlights left unchanged (char-tight; honest-framing nuance would not fit).

## Done log

- [x] (2026-05-29) Pipeline scaffold — agents created, paper/ structure, ROADMAP, TODO, refs.bib stub, main.tex skeleton, README.
- [x] (2026-05-29) Writer round 1 — §3 Background, §4.1/4.2/4.3 Method, §5.1 Setup drafted in paper/main.tex. Open `\todo{@experimenter}`: ResNet-18 main-table numbers (§5.1). Pending citations now used in body: `sensoy2018edl` (have), `miyato2018sngan`, `liu2020sngp`, `vanamersfoort2020duq`, `hendrycks2017baseline`, `yoon2024daedl` — all already on @litscout's queue.
- [x] (2026-05-29) Litscout round 1 — refs.bib populated with 17 verified core entries + 2 adjacent 2025 items; related_work_notes.md fully written with per-entry summaries and a "Risk: potential scoop" section. Two scoop risks documented (DAEDL Yoon24, F-EDL Yoon25). Naming collision discovered: "F-EDL" = Flexible EDL (Yoon25), so our method must be called "FI-EDL". I-EDL venue corrected from "Bao 2021" brief to Deng et al. 2023 ICML.
- [x] (2026-05-29) Writer round 2 — Abstract, Highlights, §1 Introduction, §2 Related Work (+ `tab:rw-concurrent` scoop-defusing comparison table), §6 Ablations (§6.1/§6.2/§6.3 with three booktabs tables). All quantitative claims trace to report §6.6 or `results/stats_significance.md`. Followed the litscout naming convention (FI-EDL vs F-EDL=Flexible EDL). Remaining writer work: §5.2 (main tables), §5.3 (score ablation), §7 Discussion, §8 Conclusion; all four are explicitly tagged as the next round.
