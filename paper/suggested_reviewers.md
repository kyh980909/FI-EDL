# Suggested Reviewers — FI-EDL Neurocomputing submission

To be sent on a separate page with the submission. **User must verify
email addresses, current affiliations, and recent collaborations before
submitting** — out-of-date entries hurt our credibility with the editor.

## Selection criteria

- **Expertise**: deep learning uncertainty (evidential, Bayesian, or
  distance-aware), calibration, or out-of-distribution detection.
- **Independence**: no direct conflict (same institution as our authors,
  recent co-authorship, or active research collaboration with the
  authors). Direct method competitors (DAEDL, F-EDL = Flexible EDL) are
  excluded as they would be obvious conflicts; they are also our most
  direct baselines so editors typically avoid them.
- **Active publication record** in the last 3 years on relevant venues
  (NeurIPS, ICML, ICLR, AAAI, journal equivalents).

## Candidates

### 1. Andrey Malinin
- Affiliation: [verify — historically Yandex / Cambridge]
- Expertise: Prior Networks (NeurIPS 2018), Dirichlet uncertainty,
  ensemble distillation. Most theoretically grounded reviewer in our
  citation graph.
- Conflict check: not a co-author of any cited FI-EDL/DAEDL/F-EDL paper.
- Why: can assess whether our diagnosis (FIM controller inertness) is
  technically correct; has authored the closest prior framework.

### 2. Joost van Amersfoort
- Affiliation: [verify — Oxford / OATML group?]
- Expertise: DUQ (ICML 2020), distance-aware deterministic uncertainty,
  spectral normalisation for OOD.
- Conflict check: independent of evidential-Dirichlet community.
- Why: most relevant non-EDL prior-art author for our SN argument; can
  evaluate §4.3 SN justification against the SNGP/DUQ literature.

### 3. Dan Hendrycks
- Affiliation: [verify — Center for AI Safety / formerly UC Berkeley]
- Expertise: the foundational OOD benchmark (ICLR 2017), CIFAR-10-C
  robustness, OOD scoring.
- Conflict check: independent of EDL community.
- Why: can assess our OOD evaluation rigor and the score-selection finding.

### 4. Mengyuan Chen
- Affiliation: [verify — Chinese Academy of Sciences / NLPR]
- Expertise: R-EDL (ICLR 2024 Spotlight), Re-EDL preprint. The closest
  "KL-free EDL" precedent.
- Conflict check: cited but not a co-author of our work.
- Why: best positioned to assess the simplification narrative — Re-EDL
  also drops the KL term, and we want a reviewer who has thought about
  why dropping KL works.

### 5. [Optional 5th candidate — user to fill]
- Possible: an editor or reviewer of the conference FI-EDL paper if their
  name is known and not a direct co-author. Or a Korean-domestic
  uncertainty-DL researcher outside the corresponding author's
  institution.

## To exclude (declared conflicts)

- Taeseong Yoon and Heeyoung Kim (authors of both DAEDL and F-EDL) —
  direct competitors and same group; editor will recognise the conflict.
- Anyone at the corresponding author's institution.
- Co-authors of the conference FI-EDL paper (user will identify and
  exclude).

## Internal note

The journal editor selects reviewers and may use none, some, or all of
our suggestions. Suggesting independent expert reviewers helps the
editor route the manuscript; it does not guarantee assignment.
