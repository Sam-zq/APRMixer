# Reproducibility boundaries

The default model is copied unchanged from the frozen plain-MSE NoCross training package. The clean model is copied unchanged from the final active architecture snapshot used for the paper figure. These names precede the APRMixer public naming.

Training and evaluation functions were extracted from the frozen plain-MSE runner with their function bodies preserved. Configuration defaults are retained, with public model naming, a portable data-root argument, station 4's existing 130 capacity, and a default of zero data workers for portability. The new launcher supports custom data paths, CPU diagnostics, source/data signature checks and evaluation-only execution. It never reuses historical checkpoints from other folders.

The original loader's physics path is retained. The unused standard-mark branch and its helper import were removed to keep the package focused on the final protocol. These presentation and launcher changes do not establish bitwise equality between full training runs on different environments. Default zero workers differs from historical four-worker launch configurations; use `--workers 4` when matching those configurations.

Synthetic tests check strict checkpoint interchange, forward parity at all four horizons, output bounds and the empirical night mask, training-only reference fitting, one-epoch CPU training, checkpoint re-evaluation and rejection of altered input data. They do not reproduce the original station results or prove scientific generalization.

The model masks night predictions internally. Training with plain all-day MSE does not remove this architectural mask. Evaluation daylight metrics use measured irradiance only after predictions have been made.

External baselines, ablation suites, sensitivity studies, hardware benchmarks and significance analyses are not included. This repository runs the APRMixer model itself. Their original experiment protocols remain separate evidence and should not be inferred from this package.

No raw data or checkpoint is included. If sharing checkpoints later, preserve the training configuration, input feature order, dataset split, fitted preprocessing context and environment records with them.
