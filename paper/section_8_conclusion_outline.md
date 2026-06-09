# §8 Conclusion — outline (hand-off for fi-edl-writer round 3)

Short section (~0.5–0.75 page, one paragraph or two).

## Structure

- **Sentence 1 — what we did**: We revisited the FI-EDL family, diagnosed the originally-proposed Fisher-information controller as empirically inert (gate λ ≈ 5×10⁻⁷ for the typical Fisher-trace scale of O(K)), and identified three components that together drive its calibration and out-of-distribution behaviour: a KL-free Brier-like fit, backbone spectral normalisation for natural-image-scale backbones, and the total-evidence score α₀ for OOD detection.

- **Sentence 2 — main quantitative result**: On CIFAR-10, the resulting recipe (FI-EDL+SN) sets a new statistically-significant SOTA on CIFAR-100 out-of-distribution AUROC (86.32 ± 0.20, p < 0.01 against each of DAEDL, F-EDL, Re-EDL), is best mean and paired-tied at the top on Expected Calibration Error (3.93 ± 0.45 vs DAEDL 4.17, p = 0.55) and on SVHN OOD AUROC (92.99 ± 1.77 vs Re-EDL 91.47, p = 0.12), and second-best on misclassification AUROC (90.70 ± 0.20).

- **Sentence 3 — honest limit**: The same recipe regresses calibration on MNIST (ECE 1.70 → 13.60 under SN); we therefore recommend SN-off for small-backbone settings and treat the SN component as task-conditional.

- **Sentence 4 — meta-message**: The work supports an Occam's-razor reading of recent EDL research — the active ingredients of strong calibration and OOD detection are surprisingly few, and adding more machinery does not always help.

- **Sentence 5 — outlook**: Open directions include per-layer Lipschitz tuning, scaling to ImageNet, hybrid with density-aware heads, and the evidential-regression analogue.

## Citations to include
- `liu2020sngp` and `vanamersfoort2020duq` in sentence 4 or outlook (positioning).
- `yoon2024daedl` and `yoon2025fedl` once each — direct competitors.
- `sensoy2018edl` — the lineage anchor.

## Tone
Confident but honest. No "always", no "the first", no "guaranteed". Use "we observe", "the recipe", "on the studied benchmarks".
