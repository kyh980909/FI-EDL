# Cover Letter Draft — Neurocomputing Submission

To be finalised by writer round 3 / user before submission. Replace all
`[BRACKET]` placeholders. Plain text below — convert to PDF/letterhead at
submission time.

---

[Date]

Editor-in-Chief
Neurocomputing
Elsevier

Dear Editor,

Please find attached our manuscript entitled

> **"Spectral-Normalized, KL-Free Evidential Deep Learning:
>  A Simpler Recipe for Calibrated Uncertainty and OOD Detection"**

for consideration as a research article in *Neurocomputing*.

## Paper summary

Evidential Deep Learning (EDL) frames classifier uncertainty as a learned
Dirichlet distribution and is widely used for calibration and
out-of-distribution (OOD) detection. Recent extensions add increasing
amounts of machinery — Fisher-information-based adaptive regularizers
(FI-EDL, our prior conference version), post-hoc Gaussian density
estimation (DAEDL, ICML 2024), or higher-flexibility distribution families
with auxiliary heads (F-EDL/Flexible EDL, NeurIPS 2025). This work asks
whether the empirical strength of these methods derives from their added
mechanisms or from a small set of simpler ingredients.

We diagnose the originally-proposed Fisher-information controller in
FI-EDL as **empirically inert** (gate weight λ ≈ 5×10⁻⁷ for the typical
Fisher-trace scale of O(K)) and identify three minimal components —
KL-free Brier-like fit, backbone spectral normalisation, and total-evidence
(α₀) as the OOD score — that together set new statistically-significant
state of the art on CIFAR-10 (CIFAR-100 OOD AUROC 86.32 ± 0.20, p<0.01
against each of DAEDL, F-EDL, Re-EDL) while removing the extra heads,
density estimators, and auxiliary regularizers added by competitors.

## Why Neurocomputing

The manuscript falls squarely within the journal's scope of theoretical
and empirical advances in neural-network learning: it combines a careful
mechanistic diagnosis of a recently-proposed regularizer family, a
simpler architectural recipe, and a methodologically rigorous comparison
against six baselines on standard benchmarks (CIFAR-10, MNIST, with
SVHN/CIFAR-100/DTD/KMNIST/FMNIST as OOD sets). The paper additionally
contributes an honest discussion of when the recipe regresses (small
backbones, MNIST scale) and the methodological observation that
per-method best-score evaluation can change reported leaderboards in
evidential-OOD literature.

## Relation to prior conference paper

This manuscript extends our conference paper FI-EDL [self-cite TODO]
along several axes:

1. Three new baselines (DAEDL, F-EDL/Flexible EDL, and a corrected I-EDL
   citation following Deng et al., ICML 2023);
2. Empirical diagnosis of the Fisher-information controller — the original
   FI-EDL paper's central mechanism — as inert at the published
   β = γ = 1 setting (~30% new content in §4.1 and §6.1–6.2);
3. Discovery of the KL-free + SN + α₀ minimal recipe, with new CIFAR-10
   SOTA on three metrics under the per-method best-score evaluation
   convention (~40% new content in §4.2, §4.3, §5, §6.3);
4. Statistical significance tests (paired t-test + 10k-resample bootstrap
   95 % confidence intervals) absent from the conference version (~5% new
   content in §5.1 + §5.2 + supplementary).

Total new content is approximately [50%] above the conference version,
which we believe meets the journal's expectations for an extension.

## Novelty position vs concurrent work

We are aware that DAEDL (Yoon & Kim, ICML 2024) was the first to apply
spectral normalisation to an EDL feature extractor and that F-EDL (Yoon
& Kim, NeurIPS 2025) independently introduces a Brier-style replacement
for the KL regulariser. Our contribution is not the individual components
but the *minimal combination* (no extra heads, no post-hoc density,
standard Dirichlet) that empirically matches or exceeds both works on
CIFAR-10 across calibration, OOD detection, and misclassification.
Section 2.2 and the head-to-head comparison in Table [tab:rw-concurrent]
make this differentiation explicit; we cite both papers prominently.

## Suggested reviewers, data/code, ethics

- A list of [3–5] suggested reviewers with affiliations and areas of
  expertise is included on a separate page.
- Code and pretrained checkpoints will be released at
  `https://github.com/[USER]/FI-EDL` upon acceptance; a private review
  link is available on request.
- All datasets used (MNIST, CIFAR-10, KMNIST, FMNIST, SVHN, CIFAR-100,
  DTD) are publicly available; no human-subjects data; no conflict of
  interest.

We confirm that this manuscript has not been published elsewhere and is
not under consideration by another journal. All authors have approved
the submission.

We look forward to your editorial decision and welcome reviewer
feedback.

Sincerely,

[Corresponding Author Name]
[Affiliation]
[Email]
[ORCID]

---

## Internal notes (delete before submission)

- Percentage of new content estimate — refine once writer round 3 has
  finalised section lengths.
- Self-citation to conference FI-EDL: user must provide bibtex key + DOI;
  add to `paper/refs.bib` and replace the `[self-cite TODO]` marker.
- Editor's name and salutation can be left as "Editor-in-Chief" if the
  current EiC is not stable; otherwise check the journal masthead.
