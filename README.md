# Ultimate AI Model Analysis Pipeline

The Ultimate AI Model Analysis Pipeline is an end-to-end framework for training, validating, and calibrating text classification models. It was designed for moderation-style datasets such as Kaggle's "Learning Agency Lab" challenges, but it also ships with synthetic data generation so the pipeline can be executed out of the box on any machine.

## Key capabilities

- **Configurable experimentation** – Centralised [`AnalysisConfig`](src/ultimate_pipeline/config.py) enables performance profiles, feature toggles, and reproducible seeding.
- **Rich text feature engineering** – [`FeatureAssembler`](src/ultimate_pipeline/features.py) combines normalisation, word- and character-level TF-IDF vectorisers, optional hashing, and dimensionality reduction via [`DimensionalityReducer`](src/ultimate_pipeline/dr.py).
- **Robust validation tooling** – [`CrossValidatorFactory`](src/ultimate_pipeline/cv.py) dynamically selects Stratified, Group, or StratifiedGroup K-Folds based on the data.
- **Advanced metrics and reporting** – [`compute_metrics`](src/ultimate_pipeline/metrics.py) tracks AUC, Brier score, log loss, and Expected Calibration Error (ECE); [`reporting`](src/ultimate_pipeline/reporting.py) persists HTML dashboards, JSON summaries, and CSV submissions.
- **Flexible calibration** – [`Calibrator`](src/ultimate_pipeline/calibration.py) supports isotonic, sigmoid, or identity calibration strategies with probability clipping safeguards.
- **Reproducible execution** – Every module respects the configured random seed, and [`data.generate_synthetic_data`](src/ultimate_pipeline/data.py) ensures deterministic fallback datasets when real data are unavailable.

## Project layout

```
configs/            # Default YAML configurations
src/ultimate_pipeline/
  cli.py            # Command line entrypoint (exposed as `ultimate-pipeline` script)
  pipeline.py       # End-to-end orchestration of feature building, CV, training, and reporting
  data.py           # Dataset discovery plus synthetic data generation
  features.py       # Text preprocessing, TF-IDF assembly, caching utilities
  models.py         # Estimator factory with LogisticRegression, SGD, and LinearSVC options
  calibration.py    # Calibration helpers used post-model training
  dr.py             # Dimensionality reduction wrapper around TruncatedSVD
  metrics.py        # Metric computations including AUC, Brier, log loss, and ECE
  reporting.py      # Run directory management and artifact persistence
```

## Quick start

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
   pip install -e .
   ```

2. **(Optional) Provide input data**
   Place `train.csv`, `test.csv`, and `sample_submission.csv` in the working directory (or a standard Kaggle mount such as `/kaggle/input`). If these files are not found the pipeline automatically generates synthetic data under `./synthetic` using the configured seed.

3. **Run the pipeline**
   ```bash
   ultimate-pipeline --performance-mode balanced --output-config runs/latest_config.json
   ```
   The CLI accepts overrides for calibration, vectorisers, dimensionality reduction, and more. Use `ultimate-pipeline --help` to see the full list of switches.

4. **Review outputs**
   Each run creates a timestamped directory inside `runs/` containing:
   - `reports/analysis_results.json` – consolidated metrics for every CV fold and overall validation performance
   - `reports/dashboard.html` – tabular dashboard of metrics and calibration diagnostics
   - `submission.csv` – averaged test-set predictions aligned with `row_id`
   - Cached feature matrices and intermediate assets (if caching is enabled)

## Configuration workflow

- Start from [`configs/default.yaml`](configs/default.yaml) or supply your own JSON/YAML file via `--config`.
- Runtime overrides supplied on the CLI (e.g. `--calibration sigmoid`, `--vectorizer hashing`) are merged on top of the selected configuration.
- The resolved configuration can be exported with `--output-config` for reproducibility.

Key toggles:

| Requirement | Configuration knobs | Implementation |
|-------------|---------------------|----------------|
| Performance profiles | `performance_mode` (`balanced`, `max_speed`, `best_accuracy`) | [`AnalysisConfig._apply_performance_mode`](src/ultimate_pipeline/config.py) |
| Calibration strategies | `calibration_enabled`, `calibration_method`, `epsilon_prob_clip` | [`Calibrator`](src/ultimate_pipeline/calibration.py) |
| Dimensionality reduction | `dimensionality_reduction`, `dimensionality_reduction_components`, `explained_variance` | [`DimensionalityReducer`](src/ultimate_pipeline/dr.py) |
| Cross-validation strategy | `n_splits_max`, dataset-derived groups | [`CrossValidatorFactory`](src/ultimate_pipeline/cv.py) |
| Feature engineering | `normalizer`, `vectorizer`, `max_tfidf_features`, TF-IDF parameter blocks | [`FeatureAssembler`](src/ultimate_pipeline/features.py) |
| Model selection | `use_sgd` or explicit `--performance-mode` | [`ModelFactory`](src/ultimate_pipeline/models.py) |

## Development tips

- Enable verbose logging by setting the `ULTIMATE_PIPELINE_LOGLEVEL=DEBUG` environment variable before running the CLI.
- When extending the pipeline, prefer updating [`AnalysisConfig`](src/ultimate_pipeline/config.py) so new capabilities are automatically exposed via configuration and CLI overrides.
- Unit tests are not bundled in this version; use `pytest` or `unittest` when adding new functionality.

## Licensing

This repository is provided for evaluation purposes. Update this section with the appropriate license text before public distribution.
