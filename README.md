# APRMixer

APRMixer combines a training-data-fitted empirical reference, a PatchMixer history encoder, a pointwise fluctuation encoder, learned reference mixing, and bounded residual correction for multistep photovoltaic power forecasting.

This repository contains the APRMixer model, empirical feature construction, standalone training and evaluation. It does not include private station data, trained checkpoints, external comparison models, or a reconstruction of every paper experiment.

## Install

Python 3.9 or newer is required. Install the PyTorch build appropriate for your GPU first, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/inference.py
```

The tests and inference example use synthetic data only. The final paper protocol uses CUDA; CPU execution is supported for diagnostics.

## Data

Place your authorized station CSV files under `datasets/` or provide another directory with `--data-root`. See [datasets/README.md](datasets/README.md) for the schema and units. Files must contain regular, ordered, unique 15-minute timestamps, five historical covariates and raw power `OT`.

Preprocessing uses a chronological 70%/10%/20% train/validation/test split (integer boundaries are defined by the loader). Covariate imputation, clipping, standardization and empirical-reference fitting use the training partition. Validation/test windows may use preceding history. Future features come from timestamps and fitted empirical rules; future observed weather is not a model input.

## Train

```bash
python train.py --data-root ./datasets --station 1 --horizon 96 --seed 2021 --device cuda:0
```

Station defaults are capacities 50, 130, 30 and 130 for stations 1–4, in the same units as the station raw power. Check these values against your own data. For a custom CSV:

```bash
python train.py --data-root ./datasets --csv my_station.csv --capacity 50 --station 1 --horizon 96 --device cuda:0
```

The frozen settings are history length 96, patch length 16, stride 8, embedding dimension 128, two Mixer blocks, batch size 32, Adam learning rate 1e-4, up to 10 epochs, and patience 3. Training uses plain all-day MSE. The checkpoint is selected using global all-day validation MSE. Prediction lengths 16/32/48/96 correspond to 4/8/12/24 hours.

For the four-station, four-horizon, three-seed APRMixer grid (48 independent runs):

```bash
python run_experiments.py --data-root ./datasets --stations 1 2 3 4 --horizons 16 32 48 96 --seeds 2021 2022 2023 --device cuda:0
```

Completed matching runs are skipped. A partially interrupted training run restarts training from its initial seed; optimizer-level continuation is not implemented. An existing output directory rejects different data, code, or configuration. Use another `--output` directory for a changed experiment.

## Evaluate

Use the same training arguments, including any custom filename, capacity, epoch count or output directory:

```bash
python evaluate.py --data-root ./datasets --station 1 --horizon 96 --seed 2021 --device cuda:0
```

Run outputs include configuration and code/data hashes, environment versions, per-epoch logs, the selected checkpoint, validation/test metrics, predictions, targets, and non-overlapping per-origin errors. Metrics include all-day and daylight errors, plus ramp metrics. Both all origins and origins spaced by the forecast horizon are reported. Measured irradiance > 1 defines the evaluation daylight mask; this evaluation-only mask differs from the empirical daylight mask used by the model.

## Two implementations

- `reproduction/APRMixer.py`: default training implementation. It preserves the historical constructor order, initialization RNG consumption, checkpoint keys and forward computation. Its archived name was CSRampFormer / NoCross. Temporary historical constructors do not leave attention or convolution modules in the final model graph.
- `models/APRMixer.py`: clean implementation for reading and inference. It contains only the active architecture. It strictly loads reproduction checkpoints and is checked for matching inference outputs at all four horizons. Its initialization consumes random numbers differently, so `--implementation clean` does not reproduce the original training initialization under the same seed.

Both use `Model(configs)(x_enc, x_mark_enc, x_dec, x_mark_dec)` and return `[batch, horizon, 1]` capacity factors. The target is the last history channel. The last six mark fields are `[envelope, daylight, normalized_envelope, sin(time), cos(time), normalized_envelope]`; the repeated final field is retained from the frozen protocol. The loader generates six additional calendar fields, giving 12 fields in total. `x_dec` is not used by the model.

## Source provenance and validation

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), [docs/source_manifest.json](docs/source_manifest.json) and [docs/VALIDATION.md](docs/VALIDATION.md). The release is source-only. Full numerical paper reproduction requires the original data, matching environment and experimental artifacts; synthetic tests do not validate the paper's accuracy claims.

## Attribution and licensing

The empirical data loader and learning-rate utility contain code adapted from Time-Series-Library. Its MIT notice is retained in [third_party/LICENSE_TSLIB](third_party/LICENSE_TSLIB). The public release deliberately omits the unused standard-calendar branch and its GluonTS-derived helper; the final empirical-harmonic path is preserved.

