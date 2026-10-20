# Changelog

## 0.2.0

A full rewrite of the package around a single `RashomonExplainer` class. The 0.1.x interface is
kept as a deprecated compatibility layer (`explainers.fis_explainer`, old plot names).

### Fixed
- `_get_all_main_effects` cached the last result by mask value only, so at loss budget 0 every
  feature after the first received the first feature's attribution.
- Score metrics (`accuracy_score`, `roc_auc_score`, `r2_score`) were treated as losses; the
  Rashomon condition ran in the wrong direction. They are now `1 - score`.
- The FIS of a sampled model mixed three models (joint effect from the pair-masked model, main
  effects from the single-masked models), producing spurious interaction for additive models.
  All three terms are now evaluated on the same joint-masked model (Eq. 9 of the paper).
- `delta != 0.1` broke the sum-to-one splits; `n_order > 2` broke the reference FIS and the FISC.
- The main-effect swarm plot coloured points by main effect instead of loss.
- `rset_explain` called `logging.shutdown()`; `config.py` created directories and reconfigured
  global logging at import time.
- `ref_explain(model_reliance=True)` ran twice and corrupted its input; `training_process` used a
  non-existent attribute; `halo_plot_3D` used matplotlib APIs removed in 3.8; seaborn `pointplot`
  arguments removed in 0.13; `log_loss` was computed on hard labels unless `predict_proba=True`.
- `loss_fn=None` raised; it now defaults to `log_loss` for classifiers and MSE otherwise.
- Saved results loaded back as lists and broke downstream indexing; `save` / `load` now restore
  the result objects.

### Changed
- Permutation attributions are batched into single model calls and the greedy search no longer
  recomputes attributions at every accepted step.
- Sampling, attributions and interactions run in parallel (`n_jobs`, joblib) with per-job seeds,
  so results are identical for any `n_jobs`.
- Results live in typed containers (`ReferenceResult`, `RashomonSamples`, `AttributionCloud`,
  `InteractionCloud`) indexed by feature index, with DataFrame summaries.
- Plots return `Axes`, accept feature names, never call `plt.show()`, and pool several explainers
  through a list argument instead of `*_multiple` variants.
- Several halo radii can be computed from one explainer (`radii=`).
- Packaging: dependencies declared in `pyproject.toml`, torch optional, build artefacts removed.

### Removed
- `fis_explainer_context` (depended on an external package) and `load_results` (old JSON layout).
