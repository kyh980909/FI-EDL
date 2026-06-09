# Review — Round 2 (2026-06-08)

Target: `papers/Neurocomputing/FI-EDL_eng.tex` (Option-B honest rewrite, 2026-06-04). First review of this version.

## Recommendation: **Minor revision**

The manuscript matured substantially since Round 1: the gate-magnitude inconsistency (R1-M1), the "below EDL baseline" factual error (R1-M2), the missing F-EDL mechanism (R1-M5), and the score-audit section (R1-M6) are properly fixed; new experiments (ResNet-18, DTD, GTSRB, CIFAR-100, TinyImageNet OOD, CIFAR-10-C) materially widen the evidence base and defuse the scoop risk. What blocks Accept is a small set of concrete, fixable defects — all writer/litscout text/bib edits, **zero new compute**: one wrong citation reused 5× in the central argument (SNGP), a stale bib that never received the Round-1 fixes, one abstract/intro claim contradicting the body, and an uncited prior-FI-EDL self-reference.

## Strengths
- Controller diagnosis is the strongest asset: mechanism-first (trigamma O(K) scale), instrumented (Table 4), reinforced by three honest negative ablations (gate-on ECE 5.19→71.62; masked-KL null; MNIST SN regression 1.70→13.60). Every gate-table number matches report §6.6.1.
- New experiments broad and verified — ResNet-18, DTD, GTSRB, CIFAR-100, TinyImageNet, CIFAR-10-C tables all match results files to reported precision. ResNet-18 ECE win (3.15, p=0.038 vs DAEDL) is a uniquely-significant result and the best addition since Round 1.
- Honest statistical framing maintained: win-iff-(p<0.05 ∧ bootstrap CI excludes 0); CIFAR-100-OOD unique-significance vs ECE/SVHN paired-tie distinction carried through abstract/§1/§5/§7. α₀ correctly downgraded to "a measurement convention, not a model contribution" (§7.2).
- Scoop largely defused via `tab:rw-concurrent` + §4.3 labeled-hypothesis mechanism (F-EDL −1.89 ECE / +2.17 C100, both verified).

## Required revisions

1. **[@litscout][HIGH] Wrong SNGP citation, reused 5×.** `Liu2020` in `fi-edl-refs.bib` (l.108-114) is "Energy-Based OOD Detection," Weitang Liu et al., NeurIPS 2020 — but the body `\citep{Liu2020}`s it as **SNGP** 5× (§1 l.115, §2.3 l.338, §4.3 l.549 & l.615, §7 l.1291), all on the load-bearing distance-awareness/bi-Lipschitz argument. SNGP is Jeremiah Liu, Lin, Padhy, Tran, Bedrax-Weiss, Lakshminarayanan, NeurIPS 2020 (arXiv 2006.10108). Add a correct `Liu2020sngp` entry, repoint all 5 cites; remove the Energy-OOD entry unless Energy-OOD is actually discussed (it is not).

2. **[@writer][HIGH] Abstract + §1 contribution 3 mis-state CIFAR-100 as an ECE "tie" with Re-EDL.** Abstract l.73 and §1 l.202-203 say "paired-tied with Re-EDL across accuracy, ECE, and four OOD directions," but `cifar100_summary.md` shows CIFAR-100 ECE vs Re-EDL is a **WIN** (Δ−20.74pp, p<0.001), matching body §5 l.983. The abstract understates a win and contradicts §5. Also ECE vs DAEDL on C100 is a documented **loss** (Δ+7.67, p<0.001, vs DAEDL's collapsed 1.76% acc). Fix to "paired-tied with Re-EDL on accuracy and four OOD directions, while improving ECE over Re-EDL," and handle the degenerate-DAEDL caveat in any "best ECE" headline.

3. **[@litscout][HIGH] Stale bibliography — Round-1 R10/R11 fixes not in canonical `fi-edl-refs.bib`.** (a) `Chen2024b` (Re-EDL) still arXiv preprint (l.185-190); published IEEE TPAMI 2025 (DOI 10.1109/TPAMI.2025.3583410) — update entry. (b) The three 2023+ refs the TODO claims were added (`liu2023cals`, `djurisic2023ash`, `liu2023gen`) are **absent**; §2.3 (l.320-352) cites nothing after 2022. Add ≥2 verified 2023-2025 refs to §2.3 + a recent EDL survey (e.g. Gao et al. 2024, arXiv 2409.04720) to §2.1.

4. **[@writer][HIGH] Prior FI-EDL conference paper referenced but never cited.** §1 l.126, §2.4 l.360, §8 l.1360 describe the gated-KL method with **no `\citep`** and no bib entry. Neurocomputing requires citing prior work + disclosing the extension relationship (TODO l.58 still open). Compounding: public "FI-EDL" is Deng et al. 2023 (ℐ-EDL)'s own abbreviation — a referee searching "FI-EDL" lands on `Deng2023`. Add the self-citation (or state it is unpublished) + one disambiguating sentence vs Deng's ℐ-EDL. The Round-1 "naming-collision footnote" was lost in the Option-B rewrite; restore it.

5. **[@writer][MED] `compute_cost.md` exists but absent from body — and it is the quantitative backbone of the "Occam's-razor / minimal" thesis.** Round-1 Q2 asked for cost; `compute_cost.md` shows FI-EDL+SN single 5.1K-param α-head vs F-EDL 270.6K-param triple head at ~equal forward time. The paper asserts "minimal/simpler" repeatedly but never reports the param/latency budget. Add a one-row-per-method cost table to §5.1 or §7.1. File device is CPU — re-run on RTX 3060 or state the CPU caveat.

6. **[@writer][MED] ResNet-18 misclass loss to DAEDL under-reported.** On ResNet-18 the loss is **−2.18pp (p=0.0003)** (`resnet_summary.md`), larger/more significant than VGG-16's −1.29pp. It is in `tab:main-resnet` (90.75 vs 92.93) but unmentioned in prose, and §7 limitation (ii) (l.1319-1324) cites only the VGG number. State the larger ResNet loss alongside it.

7. **[@writer][MED] Code/Data-availability statement thinner than R6 required.** §5.1 l.706-711 lacks GPU model (RTX 3060), framework versions, wall-clock, placeholder repo URL. Surface `CODE_RELEASE_README.md` environment details.

8. **[@writer][MED] CIFAR-10-C p-values have no source artifact.** §5.6 reports paired p-values but `cifar10c_summary.md` has only means/stds, no tests. Provide a `cifar10c_stats.md` (analogous to `stats_significance.md`) or drop p-values and report mean differences only.

9. **[@writer][LOW] §1 contribution 5 overgeneralizes α₀.** l.224 says α₀ is "the right default OOD score for standard-Dirichlet evidential methods," but `score_audit.md`/§5.4 show it is best **only** for the KL-free/density family and catastrophic for EDL/I-EDL/R-EDL/F-EDL (F-EDL −54.7pp). Align to the body's "depends on the loss family."

10. **[@writer][LOW] Highlights exceed Elsevier 3-5 max.** 6 highlights (l.91-98); merge to ≤5, each ≤85 chars.

11. **[@writer][LOW] Caption/layout nits.** (a) `fig:gate-diagnosis` caption (l.523) writes `λ(α)=exp(−tr I(α))`, dropping β,γ from Eq.(8). (b) l.1254 body prose abuts `\end{figure}` with no blank line — add paragraph break. (c) `Yoon2025` bib lacks arXiv ID (2510.18322).

## What would raise the score to Accept
Fix items 1-4 (the two wrong/stale citations, the CIFAR-100-ECE abstract contradiction, the prior-FI-EDL self-citation) — the only correctness/integrity items, all pure text/bib edits. Folding `compute_cost.md` in (5) and stating the larger ResNet misclass loss (6) converts "Occam's-razor" from asserted to demonstrated. With those, this is a clean, well-scoped, statistically careful contribution I would accept. ImageNet-100 / modern-backbone CIFAR-100-as-ID (already future work) is not required for this venue.

## Confidence: 4/5
All headline numbers cross-checked line-by-line against 11 results files; SNGP error and Re-EDL/TPAMI staleness verified against NeurIPS/IEEE/GitHub. Residual uncertainty: could not verify §5.5's claim of F-EDL-paper-reported DAEDL C100 numbers; CIFAR-10-C p-values lack a source file.
