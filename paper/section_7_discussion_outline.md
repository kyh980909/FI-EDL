# §7 Discussion — outline (hand-off for fi-edl-writer round 3)

This file sketches the talking points for §7 Discussion of the FI-EDL Neurocomputing manuscript. The writer should expand each subsection into full prose in `paper/main.tex`, replacing the `\todo{@writer: §7 Discussion}` placeholder. Every quantitative claim should cite either `results/baseline_comparison_report.md` or `results/stats_significance.md`.

## §7.1 Why does a simpler recipe match or beat more elaborate baselines?

**The skeptical reviewer question this addresses**: "If you only *remove* things and still win, what is your method actually adding?"

**Answer (three threads)**:

1. **Fewer modes of failure**.
   - F-EDL's (p, τ) sub-heads add capacity but also extra degrees of freedom that can mis-allocate on ambiguous inputs; we observe F-EDL's CIFAR-10 ECE 5.82 vs FI-EDL+SN 3.93 (Δ −1.89, p<0.001 per `results/stats_significance.md`).
   - DAEDL's post-hoc GMM is brittle when class-conditional features are not well-separated (FI-EDL+SN: VGG-16 SVHN AUROC 92.99 with α₀ score vs DAEDL 88.63, p=0.009).
   - The original FI-EDL controller introduced a regulariser that was empirically inert (§4.1) — even an "additional mechanism" that looks principled can fail to engage if its scale is mis-calibrated.

2. **Cleaner score semantics**.
   - Section §5.3 (score selection ablation) shows α₀ as an OOD score is decisively better than maxp for standard-Dirichlet methods (our FI-EDL+SN: SVHN 89.89 → 91.72 by switching maxp → α₀).
   - F-EDL uses an *effective* α_eff = α_raw + τ·p; the α₀_eff = α₀_raw + τ does not carry the same "total evidence" semantics, which we conjecture is why α₀-score on F-EDL substantially under-performs maxp-score on F-EDL.
   - We do not claim F-EDL would be improved by switching scores — that needs separate study.

3. **Backbone Lipschitz constraint is the active ingredient**.
   - Linking to SNGP \citep{liu2020sngp} and DUQ \citep{vanamersfoort2020duq}, the bi-Lipschitz backbone is the documented mechanism for distance-aware uncertainty.
   - Our experiments are consistent with this: removing SN on CIFAR-10 reduces SVHN AUROC by ~1.3 pp (see §6.3, `tab:abl-sn`), and SN by itself recovers most of DAEDL's gain over plain EDL without the explicit density head.

## §7.2 When does spectral normalisation help?

**Two regimes observed in §6.3**:

- **Natural-image scale (VGG-16, ResNet-18 on CIFAR-10)**: SN consistently helps ECE, OOD AUROC, and misclassification AUROC. ResNet-18 results from Gap 1 (`results/resnet_summary.md`) should be cited here to demonstrate backbone generality; the prediction is that the recipe ranking is preserved.
- **Small backbone / small dataset (ConvNet, MNIST)**: SN regresses ECE 1.70 → 13.60 because the Lipschitz constraint over-constrains the already-small backbone, forcing under-confidence (max prob ≈ 1/K, see §6.3 mechanism paragraph).

**Prescription**: SN is a tool, not a default. For modern large-scale natural-image classification it is essentially "free"; for small-scale benchmarks it can hurt and should be ablated per-task. This matches the broader finding in SNGP / DUQ literature that the Lipschitz tightness must be matched to the data manifold.

## §7.3 The α₀ score insight and the broader EDL evaluation convention

We re-evaluated existing EDL baselines with α₀ and observed:
- Re-EDL and DAEDL both improve at α₀ over maxp (e.g., Re-EDL CIFAR-100 84.60 → 85.82 by score swap).
- F-EDL does not improve at α₀ — a side effect of its Flexible Dirichlet (α_eff = α + τp) semantics.
- The literature should report **per-method best score** as a convention; reporting a single score across all evidential methods misrepresents weaker baselines and may also misrepresent the proposed method.

This is a methodological contribution that **does not require a new model** — purely a measurement convention.

## §7.4 Limitations

State plainly:
1. **MNIST regression under SN**: documented in §6.3. The recipe explicitly recommends SN-off for small-backbone settings, but a deployment without per-dataset ablation would risk silent regression.
2. **Single backbone family per dataset**: CIFAR-10 covers VGG-16 and ResNet-18 (Gap 1, §5.x). MNIST covers only ConvNet. ViT and stronger backbones (ResNet-50, EfficientNet) are not studied — future work.
3. **No ImageNet-scale demonstration**: compute-budget reasons. The bi-Lipschitz SN mechanism (per SNGP/DUQ) is documented at ImageNet scale in those works; our addition would be the KL-free fit + α₀ at scale.
4. **No regression EDL**: we focus on classification; the recipe analogue for evidential regression (\citealp{amini2020evidential}) is unstudied.
5. **Statistical caveats**: with 5 seeds and the headline-metric paired tests, ECE (vs DAEDL) and SVHN (vs Re-EDL/F-EDL) are paired-tied not significantly distinct. We report best-mean and explicitly mark the ties (per `results/stats_significance.md`).
6. **OOD set coverage**: SVHN + CIFAR-100 + DTD (Gap 3, `results/dtd_summary.md`) but no LSUN/iSUN/Places365 large-scale.

## §7.5 Future work

- **Learned SN coefficient** (per-layer Lipschitz tuning): may close the MNIST regression while preserving the CIFAR-10 gains.
- **Hybrid with density** (FI-EDL+SN + post-hoc density head à la DAEDL): may add the density-aware MNIST OOD lift without the calibration cost.
- **ImageNet-100 / ImageNet-1k scaling**: confirm the recipe at larger scales.
- **Regression EDL**: apply the simplification principle (drop the regularizer terms that don't engage, add SN to the backbone) to \citet{amini2020evidential} variants.
- **Calibration under corruption**: CIFAR-10-C robustness ECE/AUROC; preliminary plan in `paper/TODO.md`.

## Writing notes for the writer agent

- Target length: ~1.5–2 pages.
- Cite `liu2020sngp`, `vanamersfoort2020duq`, `miyato2018sngan` in §7.2; `amini2020evidential` in §7.5; `chen2025reedl`, `yoon2024daedl`, `yoon2025fedl` in §7.1.
- Reference `tab:rw-concurrent` (§2.2), `tab:abl-sn` (§6.3), `tab:gate` (§4.1), and the main results table once it exists (§5.2).
- Do not introduce new claims; recap and frame.
