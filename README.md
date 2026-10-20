# generalized_rashomon_set

Sample the Rashomon set around a trained model and explain **feature attributions** and
**feature interactions** across it, not just for one model.

This is the reference implementation of

* Li, Wang, Deng & Barnard, *Exploring the cloud of feature interaction scores in a Rashomon set*,
  ICLR 2024 ([arXiv:2305.10181](https://arxiv.org/abs/2305.10181)), and
* Li, Barnard & Deng, *Practical attribution guidance for Rashomon sets*
  ([arXiv:2407.18482](https://arxiv.org/abs/2407.18482)).

![FIS in the Rashomon set](https://github.com/Sichao-Li/generalized_rashomon_set/raw/main/data/FIn_Rset.png)

## What it does

A *Rashomon set* is the set of models whose loss is within `epsilon` of a reference model. This
package builds that set without retraining: each member is the reference model with one or more
input features multiplied by a mask `m`, and a greedy search finds how far each mask can move
before the loss leaves the `epsilon` band. For every sampled model it then computes permutation
based attributions:

* **main effect** `phi_i`: loss increase when feature `i` is permuted,
* **joint effect** `phi_I`: loss increase when all features in `I` are permuted,
* **feature interaction score** `FIS_I = phi_I - sum_{i in I} phi_i`.

The collection of FIS values over the Rashomon set is the *FIS cloud* (FISC). Swarm plots show
the range of each attribution across the set; halo plots show how a pair's joint loss deviates
from the no-interaction expectation.

## Install

```bash
pip install -e .            # from a clone
pip install -e ".[torch]"   # with PyTorch support
```

Requires Python >= 3.9, numpy, scikit-learn, pandas, matplotlib, seaborn and joblib.

## Quick start

```python
from sklearn.neural_network import MLPClassifier
from generalized_rashomon_set import RashomonExplainer, plots

model = MLPClassifier(max_iter=500, random_state=0).fit(X_train, y_train)

explainer = RashomonExplainer(
    model, X_test, y_test,      # X may be a DataFrame: column names are used in plots
    loss="log_loss",            # or "mse", "mae", "accuracy", "roc_auc", a callable, ...
    epsilon_rate=0.05,          # Rashomon tolerance as a fraction of the reference loss
    delta=0.1,                  # 11 loss budgets 0, 0.1 eps, ..., eps per feature and direction
    n_jobs=-1,                  # parallel over features, sampled models and feature pairs
    random_state=0,
)
explainer.fit()                 # reference -> sampling -> attributions -> interactions

explainer.attribution_summary()   # DataFrame: main-effect min / max / mean / std per feature
explainer.interaction_summary()   # DataFrame: FIS min / max / mean / std per feature pair
explainer.sampled_models()        # DataFrame: every sampled single-mask model (m, loss)

plots.swarm_plot_attribution(explainer)
plots.swarm_plot_interaction(explainer, boxplot=True)
plots.halo_plot(explainer, ("age", "priors"))     # names or indices
plots.search_trace_plot(explainer)

explainer.save("results.json")
explainer = RashomonExplainer.load("results.json", model, X_test, y_test)
```

The stages can also be run one at a time; each runs its prerequisites on demand:

```python
explainer.explain_reference(order=2)            # phi_i, phi_I, FIS_I of the reference model
explainer.sample_rashomon_set()                 # greedy mask search
explainer.explain_attributions(full_matrix=True)  # phi of every feature under every sampled model
explainer.explain_interactions(order=3, radii=(0.5, 1.0))  # triples, two halo radii
plots.halo_plot_3d(explainer, (0, 1, 2))
```

### Models

`RashomonExplainer` accepts scikit-learn estimators, statsmodels results, `torch.nn.Module`s and
anything else through `predict_fn=lambda X: ...`. The loss decides what is predicted:
`log_loss` / `roc_auc` use probabilities (`predict_proba`, or softmax/sigmoid of a torch output),
`accuracy` uses labels, regression losses use `predict`. Score metrics (`accuracy`, `roc_auc`,
`r2`) are converted to losses (`1 - score`) so the Rashomon condition is always "loss within
epsilon". For torch models use `backend="threading"`.

### Images

Super-pixels can be features: pass `feature_space=ImageSegmentFeatureSpace(segments)` with `X`
of shape `(n, H, W, C)` and a `predict_fn` that runs your vision model.

### Main parameters

| parameter | meaning |
|---|---|
| `epsilon_rate` / `epsilon` | Rashomon tolerance, relative to the reference loss or absolute |
| `delta` | resolution of the loss budget grid (must divide 1); `0.1` gives 11 models per feature and direction |
| `features`, `subsets` | restrict the analysis to some features / feature subsets |
| `shuffle_times` | permutations averaged per attribution (`interaction_shuffle_times` for the FISC stage) |
| `permutation` | `"independent"` (default) or `"joint"` permutation of a feature subset |
| `drop_insensitive` | skip features whose reference main effect is exactly 0 |
| `n_jobs`, `backend`, `batch_size` | parallelism (joblib) and the maximum rows per model call |
| `lr`, `max_refinements`, `patience`, `m_max` | greedy-search controls (step, step refinements, stall detection, mask bound) |

### Reproducibility and speed

Every attribution estimate has its own seed derived from `random_state`, so results are identical
for any `n_jobs` and backend. Permutations are batched into single model calls, the greedy search
no longer recomputes attributions at every step, and the three expensive stages are parallel:
sampling over `(feature, direction)`, attributions over sampled models, interactions over feature
subsets.

## Demos

* [demo/quickstart.ipynb](demo/quickstart.ipynb): synthetic regression with one true interaction, all plots, save/load.
* [demo/toy_example.ipynb](demo/toy_example.ipynb): the halo plot of `x1 + x2 + x1*x2` at one point, analytic (sympy) versus the explainer.
* [experiments/recidivism prediction.ipynb](experiments/recidivism%20prediction.ipynb): the COMPAS experiment of the paper on the 0.2 API.

## Upgrading from 0.1.x

`explainers.fis_explainer`, `ref_explain()`, `rset_explain()` and the old plot names
(`swarm_plot_FIS`, `swarm_plot_MR`, `halo_plot_multiple`, ...) still work and emit
`DeprecationWarning`s; the old result attributes (`ref_analysis`, `rset_main_effect_processed`,
`fis_in_r`, `loss_in_r`, `all_pairs`, `v_list`) are filled in. See `CHANGELOG.md` for the
behavioural fixes, in particular: FIS of a sampled model is now evaluated entirely on that model
(the 0.1.x code mixed the joint-masked model with the single-masked ones), score metrics are
minimised correctly, and `delta != 0.1` and `order > 2` work.

## Project layout

```
src/generalized_rashomon_set/
  _explainer.py       RashomonExplainer (pipeline, summaries, save/load)
  _search.py          greedy mask search (Algorithm 1)
  _attribution.py     batched permutation attributions + parallel workers
  _loss.py            loss registry (Loss)          _predictor.py   model adapter (Predictor)
  _feature_space.py   tabular / image-segment masking and permutation
  _combinatorics.py   subsets, loss-budget grid, sum-to-one splits
  _results.py         ReferenceResult, RashomonSamples, AttributionCloud, InteractionCloud
  plots/              swarm_plot_attribution, swarm_plot_interaction, halo_plot, halo_plot_3d, search_trace_plot
  explainers/, utils/, config.py   0.1.x compatibility layer
tests/                pytest suite (synthetic data + COMPAS)
demo/, experiments/   notebooks
```

## Citation

```
@inproceedings{li2024exploring,
  title={Exploring the cloud of feature interaction scores in a Rashomon set},
  author={Li, Sichao and Wang, Rong and Deng, Quanling and Barnard, Amanda},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

Questions: [Sichao Li](mailto:sichao.li@anu.edu.au).
