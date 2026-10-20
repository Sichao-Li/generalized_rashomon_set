# Recidivism experiment: 0.2 code vs. the paper

This note compares the output of `recidivism prediction.ipynb` (package version 0.2.0, executed on
2026-10-20) with the COMPAS results reported in *Exploring the cloud of feature interaction scores in
a Rashomon set* (ICLR 2024, [arXiv:2305.10181v2](https://arxiv.org/html/2305.10181v2)) and with the
figures embedded in the 2023 version of the same notebook (git history, commit `b634da5`). All numbers
below are copied from the executed notebook; the figures it saves are in `figures/`.

## Setup

| | Notebook (0.2) | Paper / 2023 notebook |
|---|---|---|
| Data | `data/Recidivism prediction/compas.csv`, 7 214 rows, 6 binary features + intercept | same |
| Split | `train_test_split(random_state=2020, test_size=0.3)`; explanations on the test split | same code |
| Reference model | `statsmodels.Logit` fitted on the training split (train log-loss 0.6401) | same |
| VIC model | same `Logit` with the coefficients published with VIC; train log-loss 0.6403 | the 2023 notebook printed 0.7277 because `log_loss` was computed on hard labels (see `CHANGELOG.md`) |
| Loss | log-loss on predicted probabilities | log-loss on hard labels in the 2023 code |
| `epsilon_rate` | 0.02, 0.04, 0.06, 0.08, 0.10 (halo and swarm plots); 0.05 (VIC comparison) | same |
| `delta` | 0.1 | same |
| FIS of a sampled model | all three terms of Eq. 9 evaluated on the joint-masked model | joint effect from the joint-masked model, main effects from the two single-masked models |

The last row is the deliberate behavioural change documented in `CHANGELOG.md` and is the expected
source of most quantitative differences below. Two further sources: the loss (and therefore
`epsilon = epsilon_rate * loss_ref`) is now computed on probabilities instead of hard labels, so every
Rashomon set is defined on a different absolute scale, and the permutation attributions use different
random shuffles.

Note on figure numbering: the brief that produced this note referred to "Figure 2 halo plots". In the
arXiv version, Figure 2 shows halo plots of the synthetic functions of Appendix F; the COMPAS halo
plots are Figure 4 (pairwise `prior` x `juvenile`, and the 3-D halo of `prior`, `juvenile`,
`charge`), the COMPAS swarm plot is Figure 5(b), and the model-reliance comparison with VIC is Table 2.
The sections below use the paper's numbering.

## Figure 4 (top): pairwise halo plot

Saved: `figures/halo_prior_gender_epsilon.png` (the pair requested for this run) and
`figures/halo_prior_juvenilecrime_epsilon.png` (the pair shown in the paper). Each figure draws the
five explainers (`epsilon_rate` 0.02 to 0.10) as one ring each, in units of their own `epsilon`.

What the new code shows. For both pairs the empirical halo clearly departs from the expected circle at
every `epsilon`, i.e. the two features interact. The halo radius in units of `epsilon`
(notebook table "extent of the empirical halo"):

| `epsilon_rate` | pair | max radius / eps | min radius / eps | mean radius, quadrants I and III | mean radius, quadrants II and IV |
|---|---|---|---|---|---|
| 0.02 | prior x gender | 1.59 | 0.36 | 0.61 | 1.38 |
| 0.10 | prior x gender | 1.38 | 0.49 | 0.64 | 1.32 |
| 0.02 | prior x juvenilecrime | 1.68 | 0.25 | 1.45 | 0.53 |
| 0.10 | prior x juvenilecrime | 1.48 | 0.37 | 1.39 | 0.56 |

So `prior x juvenilecrime` is elongated along the main diagonal (the joint loss exceeds the sum of the
single losses when both masks move the loss in the same direction, by about 40 to 45 %, and falls
short of it by about 45 % when they move in opposite directions). `prior x gender` shows the mirror
pattern, elongated along the anti-diagonal. Relative to `epsilon` the shape is almost scale-free: the
rings for the five rates nearly coincide, with the outermost radius shrinking slowly from 1.6 to 1.4
`epsilon` as the rate grows. In absolute loss units the halo therefore grows linearly with `epsilon`.

Comparison with the paper. The 2023 notebook plotted the same pair as the paper (`pair_idx=16` is
`(prior, juvenilecrime)` under `itertools.combinations(range(7), 2)`), and its halo (cell 18 of the
2023 notebook) is elongated along the negative `prior` axis and symmetric in the `juvenile` direction,
not along a diagonal. The qualitative conclusion of the paper, that the joint loss of this pair is
not the sum of the single-feature losses (the pair interacts) and that the interaction can push a
model out of the Rashomon set, holds in the new run. The shape of the halo does not reproduce; this is
expected from the Eq. 9 change because the quadrant pattern depends on which model's main effects are
subtracted. The paper's statement that the halos for different `t` are nested and grow with the budget
also holds (see the single-explainer plot with `radii=(0.2, ..., 1.0)` in the notebook).

## Figure 5(b): swarm plot of the FISC

Saved: `figures/swarm_fis_pairs.png` (signed FIS, `epsilon_rate=0.10`) and
`figures/swarm_fis_pairs_epsilon.png` (absolute FIS pooled over the five rates, which is the form of
the paper's Figure 5(b) and of cell 21 of the 2023 notebook). Pairs: `age x prior`, `race x prior`,
`prior x gender`, `prior x juvenilecrime`, `prior x currentcharge`.

Reference FIS and FIS range (max minus min over the 36 joint-masked models at the full budget) from
the notebook table `fis_by_epsilon`:

| pair | reference FIS | range at 0.02 | range at 0.04 | range at 0.06 | range at 0.08 | range at 0.10 |
|---|---|---|---|---|---|---|
| age x prior | -0.0008 | 0.0086 | 0.0122 | 0.0150 | 0.0173 | 0.0193 |
| race x prior | 0.0036 | 0.0141 | 0.0181 | 0.0213 | 0.0240 | 0.0265 |
| prior x gender | 0.0016 | 0.0078 | 0.0097 | 0.0113 | 0.0127 | 0.0141 |
| prior x juvenilecrime | 0.0073 | 0.0131 | 0.0160 | 0.0188 | 0.0212 | 0.0238 |
| prior x currentcharge | 0.0025 | 0.0125 | 0.0170 | 0.0208 | 0.0240 | 0.0270 |

Values read off the paper's Figure 5(b) (absolute FIS, pooled over the five rates): reference FIS
about 0.004 for `prior x juvenilecrime`, 0.002 for `prior x gender`, below 0.001 for the other three;
largest observed |FIS| about 0.011 (`age x prior`), 0.010 (`prior x juvenilecrime`), 0.009
(`prior x gender`), 0.0075 (`prior x currentcharge`), 0.0035 (`race x prior`).

What holds:

- `prior x juvenilecrime` has the largest reference FIS in both runs (0.0073 now, about 0.004 in the
  paper) and the widest box in the pooled absolute plot (largest |FIS| 0.022 now, 0.010 in the paper).
- The FIS range grows monotonically with `epsilon` for every pair. The growth is sub-linear: a
  five-fold `epsilon` widens the range by a factor 1.8 (`prior x gender`) to 2.2
  (`prior x currentcharge`, `age x prior`).
- For every pair the FISC contains both signs, and the reference FIS sits well inside the cloud, which
  is the paper's main point about the FISC.
- Exactly half of the joint-masked models at the full budget lie outside the Rashomon set
  (`frac_in_set = 0.5` for all pairs and all rates): the interaction pushes the joint model past the
  boundary in two of the four sign quadrants, as the halo plots show.

What does not hold:

- The ordering of the other pairs changes. In the paper `prior x gender` is the second-widest cloud
  and `race x prior` the narrowest; in the new run `prior x gender` is the narrowest of the five
  (range 0.014 at `epsilon_rate=0.10`) and `race x prior` is the second widest (0.027). The new run
  does not show `prior x gender` as a dominant interaction; its reference FIS (0.0016) is the
  second smallest of the five pairs.
- Absolute FIS values are about twice those of the paper at the same `epsilon_rate`. Part of this is
  the Eq. 9 change, part the different loss scale (log-loss on probabilities).

## Table 2: model reliance compared with VIC

The notebook computes, for the VIC coefficient model at `epsilon_rate=0.05`, the model reliance
ratio `loss_permuted / loss_model` of every sampled model (`AttributionCloud.ratios`, 22 models per
feature):

| feature | MR max | MR min | MR range | MR mean | paper (greedy) range | VIC range |
|---|---|---|---|---|---|---|
| age | 1.098 | 0.977 | 0.121 | 1.028 | 0.068 | 0.073 |
| race | 1.048 | 0.983 | 0.065 | 1.016 | 0.088 | 0.043 |
| prior | 1.110 | 0.992 | 0.117 | 1.050 | 0.249 | 0.129 |
| gender | 1.014 | 0.994 | 0.020 | 1.004 | 0.086 | 0.043 |
| juvenilecrime | 1.055 | 1.014 | 0.041 | 1.035 | 0.112 | 0.105 |
| currentcharge | 1.041 | 0.986 | 0.055 | 1.012 | 0.091 | 0.031 |

The two right-hand columns are `(MCR+) - (MCR-)` from Table 2 of the paper (greedy search in bold,
VIC in the unbolded column).

What holds: `prior` and `juvenilecrime` have the largest mean reliance in both runs (1.050 and 1.035
now; 1.330 and 1.367 in the paper), and `prior` has one of the two widest ranges.

What does not hold, or is not comparable:

- The magnitudes differ by an order of magnitude in the ranges and the mean reliances are much closer
  to 1. The bold numbers of Table 2 were produced by the 2023 notebook with a different estimator: it
  replaced the binary feature by 1 and by -1 (the data are 0/1 coded) weighted by the frequency of 1s,
  evaluated that frequency on the masked column (so it was 0 for every mask other than 1), and applied
  the replacement to the reference model rather than to the sampled model. The 0.2 code uses the
  permutation estimator of the paper's main effect (Section 3) instead. The two columns are therefore different
  quantities and the table above cannot confirm or refute the paper's claim that the greedy search
  finds a wider reliance range than VIC. With the permutation estimator, the ranges found by the
  greedy search are of the same size as the VIC ranges (0.12 vs 0.13 for `prior`, 0.12 vs 0.07 for
  `age`, 0.04 vs 0.11 for `juvenilecrime`).
- `age` now has the widest range (0.121), slightly above `prior` (0.117); in the paper `prior` was
  far ahead (0.249 vs 0.068).

The VIC coefficient model reaches a training log-loss of 0.6403 against 0.6401 for the refitted
logistic regression, so on this split it is within 0.04 % of the optimum. Appendix G of the paper
states that the VIC model "can be optimized further", which the 2023 notebook's hard-label loss suggested (it printed 0.728 for
the VIC model) but which is marginal for the log-loss on probabilities.

## Figure 4 (bottom): third-order interactions and the 3-D halo

Saved: `figures/halo3d_prior_gender_juvenilecrime.png`. The notebook draws the triple
(`prior`, `gender`, `juvenilecrime`); the paper draws (`prior`, `juvenile`, `charge`). The third-order
summary of the new run ranks `race x prior x juvenilecrime` (reference FIS 0.0105),
`prior x juvenilecrime x currentcharge` (0.0081) and `prior x gender x juvenilecrime` (0.0075) highest;
the paper gives no numbers for third-order FIS, so only the existence of a non-trivial 3-D halo can be
compared, and that holds.

## Attribution swarm plot

Saved: `figures/swarm_attribution.png` (`epsilon_rate=0.10`, refitted model). The paper has no
corresponding figure for COMPAS. `prior` has the largest reference main effect (0.030) and together
with `age` the widest cloud (ranges 0.079 and 0.081); `gender` has the smallest (0.013). Several
sampled models have a negative main effect for `age`, `race`, `prior`, `gender` and `currentcharge`,
i.e. permuting the feature lowers the test loss of that model.

## Demo notebooks

`demo/x1x2_simple.png` was regenerated from `demo/toy_example.ipynb` (Appendix F of the paper). The
greedy search recovers the analytic masks to 8.8e-5 and the explainer's empirical halo coincides with
the analytic one; for the additive control model the empirical halo is within 3 % of `epsilon` of the
expected circle. `demo/quickstart.ipynb` ran without changes; on the synthetic regression with one
true interaction the reference FIS of `x0 x x1` is -6.79 against values below 0.2 in magnitude for the
other pairs.
