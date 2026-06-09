# Related Work Notes

Maintained by `fi-edl-litscout`. Each entry: claim, method, why we cite it.
Append after reading at least the abstract. Never invent.
Last update: 2026-05-29.

## EDL lineage

### Sensoy, Kaplan, Kandemir 2018 NeurIPS — EDL (`sensoy2018edl`)
Founding paper of evidential deep learning for classification. Places a
Dirichlet prior over class probabilities and trains a deterministic
network to output evidence (`alpha = evidence + 1`) using a Brier
(sum-of-squares) loss plus a KL-to-uniform regularizer on the
misclassified-class evidence. Our paper cites this as the canonical
formulation we diagnose and modify: the FI-EDL recipe drops the
KL-to-uniform regularizer and re-derives a Fisher-information-aware
training criterion, so we need to clearly contrast against the original
loss.

### Malinin & Gales 2018 NeurIPS — Prior Networks (`malinin2018prior`)
Parallel line of work that explicitly models distributional uncertainty
with a Dirichlet output but trains by minimizing KL between predicted
Dirichlet and a target Dirichlet built from in-distribution (sharp) and
synthetic out-of-distribution (flat) samples. We cite it in §2 as the
"OOD-supervised" sibling of EDL, and to motivate why we instead pursue
an OOD-free training recipe.

### Amini, Schwarting, Soleimany, Rus 2020 NeurIPS — Deep Evidential Regression (`amini2020evidential`)
Extends the evidential framework to regression by placing a
Normal-Inverse-Gamma prior on the Gaussian likelihood, training a single
network to predict the four hyperparameters, and reading aleatoric and
epistemic uncertainty from closed-form moments. We cite this as the
regression analogue, mainly to position FI-EDL as a classification-side
contribution and acknowledge the broader evidential program.

### Deng, Chen, Yu, Liu, Heng 2023 ICML — I-EDL / Fisher-information EDL (`deng2023iedl`)
**Important venue/year correction:** the original task brief said
"Bao et al. 2021"; the actual paper is Deng et al., ICML 2023
(PMLR v202:7596–7616, arXiv:2303.02045). Introduces the Fisher
Information Matrix of the Dirichlet to reweight EDL's per-class loss
terms so that uncertain (low-evidence) classes receive more attention,
yielding better calibration and few-shot performance. **Highly relevant
to us**: FI-EDL also leverages Fisher information of the Dirichlet, so
this is our nearest theoretical neighbor and must be cited and
differentiated carefully in §2 and §4.

### Chen, Gao, Xu 2024 ICLR Spotlight — R-EDL (`chen2024redl`)
Identifies two "nonessential" choices in EDL: the fixed prior weight
(equal to the number of classes) and the variance-minimizing
regularization term. R-EDL treats the prior weight as an adjustable
hyperparameter and directly optimizes the expectation of the Dirichlet
PDF. We cite it as direct evidence from the community that the prior
weight (`alpha_0`) is a tunable knob — supporting FI-EDL's alpha0
treatment as principled rather than ad hoc.

### Chen, Gao, Xu 2025 TPAMI — Re-EDL (journal extension) (`chen2025reedl`)
Journal extension of R-EDL that goes further by also dropping the
KL-divergence regularization term entirely. This is the closest
KL-free EDL baseline to our recipe. We cite it as our primary baseline
and as community precedent for the "KL-free is OK and often better"
claim that anchors our diagnosis. **Update 2026-05-29:** accepted at
IEEE TPAMI 2025 (formerly arXiv:2410.00393); bibkey renamed from
`chen2024reedl` to `chen2025reedl`. No "preprint" caveat needed at
first citation.

### Yoon & Kim 2024 ICML — DAEDL (`yoon2024daedl`) — POTENTIAL SCOOP
Density-Aware EDL. Adds a feature-space density estimator (Gaussian
Discriminant Analysis at test time) on top of EDL and — crucially —
**applies spectral normalization to the feature extractor** so the
feature distance is bounded by the input distance, making the GDA
density meaningful. They write: *"DAEDL employs spectral normalization
(Miyato et al., 2018) in the feature extractor f_θ ... we are the first
to adopt it for DBU models."* This is the first published combination
of EDL + spectral normalization for image classification. See "Risk:
potential scoop" below.

### Yoon & Kim 2025 NeurIPS — F-EDL / Flexible EDL (`yoon2025fedl`) — POTENTIAL SCOOP
Extends EDL by replacing the Dirichlet with a Flexible Dirichlet (FD)
that captures multimodal/mixture beliefs and learns input-dependent
improper priors. Importantly for us, F-EDL **also applies spectral
normalization to f_θ and g_φ1** and **replaces the KL penalty with a
Brier-score-based regularizer** on the predictive mean. This is the
most direct overlap with FI-EDL's "SN + drop-KL" recipe and must be
addressed head-on. See "Risk: potential scoop" below.

**Naming note for the writer:** in earlier internal notes the project
sometimes used "F-EDL" to mean "Fisher-information EDL". The published
2025 NeurIPS paper uses "F-EDL = Flexible EDL". To avoid collision,
we should either rename our method or always spell out "FI-EDL"
(Fisher-Information EDL) and never abbreviate to F-EDL in the
manuscript.

## Calibration

### Guo, Pleiss, Sun, Weinberger 2017 ICML (`guo2017calibration`)
The reference for the modern miscalibration story: deeper/wider
networks with BN are systematically overconfident, and a one-parameter
temperature-scaling post-hoc fix recovers most of the lost calibration
on standard benchmarks. We cite it as the source of (a) the ECE metric
we report and (b) temperature scaling as a post-hoc calibrator we
compare against and stack on top of FI-EDL.

### Mukhoti, Kulharia, Sanyal, Golodetz, Torr, Dokania 2020 NeurIPS (`mukhoti2020focal`)
Shows that training with focal loss instead of cross-entropy yields
networks that are already well-calibrated, and that focal+temperature
scaling reaches state-of-the-art calibration without sacrificing
accuracy. We cite it as the strongest "train-time calibration" baseline
and to contextualize our claim that FI-EDL achieves calibration at
training time without focal loss.

### Liu, Rony, Galdran, Dolz, Ben Ayed 2023 CVPR — CALS (`liu2023cals`)
Class-Adaptive Label Smoothing. Generalises uniform label smoothing by
learning per-class multipliers via an Augmented-Lagrangian penalty with
an inequality margin constraint on logit distances, yielding a
constrained-optimisation view of margin-based calibration. Reports
state-of-the-art calibration on ImageNet, ImageNet-LT, and semantic
segmentation. We cite this as the **2023+ train-time calibration
reference** required by round-1 m6 (§2.3 had no work after 2022),
positioning FI-EDL as a Dirichlet-output alternative to logit-margin
calibration that arrives at calibration through evidence formation
rather than logit shaping.

## OOD detection

### Hendrycks & Gimpel 2017 ICLR — MSP baseline (`hendrycks2017baseline`)
Establishes the standard OOD-detection protocol: rank in-distribution
vs OOD by the maximum softmax probability (MSP) and report AUROC,
AUPR, FPR@95. We cite it for (a) the MSP baseline number in our OOD
tables and (b) the evaluation protocol we adopt.

### Liang, Li, Srikant 2018 ICLR — ODIN (`liang2018odin`)
Improves MSP-based OOD detection by combining temperature scaling and
small input perturbations along the gradient direction of the predicted
class. We cite it as a post-hoc OOD baseline and to motivate
*train-time* OOD-aware methods like EDL that do not require an extra
backward pass at inference.

### Lee, Lee, Lee, Shin 2018 NeurIPS — Mahalanobis (`lee2018mahalanobis`)
Class-conditional Gaussian fit in the feature space, with the
Mahalanobis distance to the closest class centroid used as an OOD
score, optionally combined with input perturbation. We cite it as a
strong feature-space OOD baseline and as the spiritual ancestor of
DAEDL's density-based score.

### Liu et al. 2020 NeurIPS — SNGP (`liu2020sngp`) — VERY RELEVANT
Spectral-normalized Neural Gaussian Process. Combines (a) spectral
normalization on the hidden layers to enforce a bi-Lipschitz mapping
that is "distance-aware" and (b) a Gaussian-process output layer with
Laplace approximation. This is the most important "SN-for-uncertainty"
reference for our paper. We cite it as theoretical justification for
why spectral normalization helps distance-aware uncertainty
estimation and as a deterministic-single-pass uncertainty baseline.

### Sun, Ming, Zhu, Li 2022 ICML — KNN-OOD (`sun2022knn`)
Non-parametric OOD detection using the distance to the k-th nearest
training feature in a normalized embedding space. We cite it as a
recent strong non-parametric baseline and to support the framing that
"good features → good OOD" — which is exactly what spectral
normalization is meant to provide.

### Djurisic, Bozanic, Ashok, Liu 2023 ICLR — ASH (`djurisic2023ash`)
Activation Shaping. A one-line, inference-only post-hoc OOD detector
that prunes roughly 90% of a sample's late-layer activations and
lightly rescales the survivors, then uses any standard score (MSP,
energy, etc.) on the shaped activations. Achieves SOTA post-hoc
ImageNet OOD detection without training-time intervention or
training-statistics. We cite it as the **2023 post-hoc OOD reference**
required by round-1 m6 and as a representative of the
"feature-shaping" family that is conceptually orthogonal to FI-EDL's
training-time evidence formation — the two could be stacked but our
paper does not.

### Liu, Lochman, Zach 2023 CVPR — GEN (`liu2023gen`)
Generalised Entropy score. Replaces Shannon entropy on softmax outputs
with a generalised entropy that is more sensitive to sub-unit
probabilities; applied post-hoc to any softmax classifier, it yields
+3.5% average AUROC on ImageNet-1k OOD benchmarks. We cite it as a
recent softmax-based post-hoc OOD baseline that is orthogonal to
FI-EDL's evidential $\alpha_0$ score (different output family);
together with ASH it covers the 2023 post-hoc OOD landscape that §2.3
previously omitted.

## Spectral normalization & Lipschitz

### Miyato, Kataoka, Koyama, Yoshida 2018 ICLR — SN-GAN (`miyato2018sngan`)
Introduces the power-iteration-based spectral-norm constraint on each
weight matrix (originally to stabilize GAN training) and shows it
yields a 1-Lipschitz layer. This is the algorithm we use in our recipe;
we cite it whenever we introduce SN technically.

### van Amersfoort, Smith, Teh, Gal 2020 ICML — DUQ (`vanamersfoort2020duq`)
Deterministic-uncertainty model based on RBF centroids in a learned
feature space with a two-sided gradient penalty for bi-Lipschitz
sensitivity. We cite it as the canonical example that "Lipschitz +
distance ≈ uncertainty" works for a single deterministic network, even
without a GP head — giving cover to the FI-EDL+SN combination.

### Bartlett, Foster, Telgarsky 2017 NeurIPS — spectrally-normalized margin bounds (`bartlett2017spectrally`)
Generalization bound for neural networks that scales with the product
of spectral norms (Lipschitz constant) divided by margins. We cite it
as the theoretical motivation for why constraining spectral norms is
expected to improve generalization and calibration, not just OOD
distance-awareness.

## Risk: potential scoop (2024–2026)

The following two papers are the **closest existing overlap** with
FI-EDL's recipe and should be explicitly differentiated in §2 of the
manuscript.

### 1. DAEDL — Yoon & Kim, ICML 2024 (`yoon2024daedl`)
**Overlap:** First to apply spectral normalization on top of an EDL
feature extractor. Uses GDA density at test time. Keeps the standard
EDL loss (KL regularizer retained).
**Difference vs FI-EDL:** DAEDL needs the extra GDA density head and
does *not* modify the EDL training loss. FI-EDL targets the *loss*:
diagnoses the KL term as harmful, replaces it with a Fisher-information
formulation, and treats `alpha_0` as a tunable knob. Spectral
normalization in FI-EDL is a regularizer of the Lipschitz constant
that supports our theoretical claim, not a prerequisite for a
post-hoc density model.
**Action for writer:** explicitly position FI-EDL as "training-time"
versus DAEDL's "test-time density on top". The two are complementary
in principle.

### 2. F-EDL (Flexible EDL) — Yoon & Kim, NeurIPS 2025 (`yoon2025fedl`)
**Overlap:** *Largest overlap so far.* F-EDL (a) applies spectral
normalization to the feature extractor and the auxiliary head,
(b) replaces the KL penalty with a Brier-score-based regularizer on
the predictive mean, and (c) introduces input-dependent priors.
Three of our four design ingredients (SN, drop-KL, prior tuning) are
present.
**Difference vs FI-EDL:** F-EDL changes the *distribution family*
(Flexible Dirichlet, not Dirichlet) and uses Brier-on-mean as the
replacement regularizer. FI-EDL keeps the Dirichlet and replaces the
KL term with a Fisher-information-derived criterion that has a clean
theoretical interpretation as a local volume penalty. We must (a)
acknowledge F-EDL as concurrent work, (b) emphasize that the Fisher
derivation is novel and distinct, and (c) include F-EDL as the
strongest baseline in our experiments — which the codebase already
does (see `src/models/heads/daedl_head.py` and the F-EDL configs).
**Action for writer:** lead §2 with a "concurrent work" subsection
that places F-EDL and DAEDL side-by-side with our recipe in a small
table (SN ✓/✗, KL dropped ✓/✗, prior tuned ✓/✗, mechanism).
**Action for naming:** our paper must not abbreviate "Fisher-
Information EDL" as "F-EDL" because that abbreviation now belongs to
Yoon & Kim 2025. Use "FI-EDL" consistently.

### 3. GEM-FI — Mohammed, Daneshfar, Liò, ICML 2026 (`mohammed2026gemfi`) — NEW (discovered 2026-05-29)
**Overlap:** Title-level Fisher + Evidential overlap is the strongest of
any 2026 paper I have found this round. GEM-FI introduces a *Fisher-
informed regularizer* on a mixture-of-evidential-heads architecture,
gated by an energy signal, and reports calibration / ID-OOD-separation
gains with single-pass inference. The naming "Fisher modulation" sits
uncomfortably close to "Fisher-Information EDL".
**Difference vs FI-EDL:** GEM-FI's Fisher term acts as an *allocator
regulariser on the mixture weights* to prevent head collapse — it does
**not** weight the per-class loss by Fisher information of the
Dirichlet (which is the Deng-Chen 2023 ICML I-EDL formulation that
FI-EDL inherits and diagnoses). Architecturally GEM-FI requires
multiple evidential heads and an energy-gating module; FI-EDL has a
single head with no gating. No spectral normalization in GEM-FI.
**Action for writer:** add one sentence in §2 distinguishing FI-EDL
from GEM-FI ("Concurrent ICML 2026 work GEM-FI also uses Fisher in an
evidential context, but as a mixture-allocation regulariser rather
than a per-sample loss weight"). Cite as `mohammed2026gemfi`. Do
**not** spend an entire row in `tab:rw-concurrent` on it — GEM-FI does
not use SN, KL-drop, or alpha0, so the head-to-head comparison would
be one-sided. A footnote is sufficient.
**Action for naming:** the title-level collision with our "FI" is
unfortunate but the methodological diff is large enough that confusion
will be limited if §2 is explicit.

### 4. Other 2025 arXiv papers (lower-risk, domain-specific)
- *Evidential Spectrum-Aware Contrastive Learning for OOD Detection in
  Dynamic Graphs* (Sun, Lin, Zhou, Shang, Cheng, Cao; ECML-PKDD 2025;
  arXiv:2506.07417) — graph-domain; uses "spectrum-aware" contrastive
  augmentation, not spectral normalization. Cited as `sun2025evisec`
  for completeness. **Update 2026-05-29:** Anonymous placeholder
  replaced with verified author list; ECML-PKDD 2025 acceptance
  confirmed via arXiv landing-page note.
- *Open-Set Domain Generalization through Spectral-Spatial Uncertainty
  Disentanglement for Hyperspectral Image Classification*
  (arXiv:2506.09460) — hyperspectral imaging; "spectral" here refers
  to wavelength bands, not spectral norms. Cited as
  `khoshbakht2025hyperspectral` for completeness.

Neither domain-specific paper threatens scoop on the FI-EDL +
classification recipe.

## Follow-ups discovered

- [x] (2026-05-29) Verified the EviSEC (`sun2025evisec`) author list
      from arXiv 2506.07417 — Sun, Lin, Zhou, Shang, Cheng, Cao —
      and confirmed ECML-PKDD 2025 acceptance. Bibkey renamed from
      `evisec2025` to `sun2025evisec`.
- [x] (2026-05-29) Verified Re-EDL venue — accepted at TPAMI 2025
      (IEEE Xplore document 11052867; confirmed via official authors'
      GitHub repo header). Bibkey renamed from `chen2024reedl` to
      `chen2025reedl`; main.tex citation keys updated by `sed`.
- [ ] Decide on a consistent abbreviation for our method ("FI-EDL"
      strongly preferred over "F-EDL" because of the F-EDL naming
      collision with Yoon & Kim 2025 NeurIPS).
- [ ] After the first §2 draft, run a follow-up scoop scan for any
      NeurIPS-2026 / ICLR-2026 submissions that may have appeared
      since 2026-05-29.
- [ ] Consider citing Krishnan & Tickoo 2020 NeurIPS (mixup-based
      calibration) if §2 needs broader calibration coverage —
      deferred for now.
- [ ] Consider citing Behrmann et al. 2019 ICML (invertible ResNet)
      if §4 needs a deeper Lipschitz-architecture reference —
      deferred for now.
