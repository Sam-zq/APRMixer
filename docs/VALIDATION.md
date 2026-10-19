# Local validation

Checked on 2026-10-14 with Python 3.9, PyTorch 1.12.1, NumPy 1.23.5, pandas 1.5.3 and scikit-learn 1.2.1. Diagnostics ran on CPU; no full real-data training or CUDA reproducibility claim is made.

Command:

```bash
python -m unittest discover -s tests -v
```

All three tests passed:

1. Strict reproduction-to-clean checkpoint loading and matching inference at H=16/32/48/96, finite bounded outputs, empirical night zeros, and no active attention module in the final reproduction model.
2. Training data and all timestamp/reference marks stay unchanged when only validation/test observations are changed; test targets do change.
3. A one-epoch synthetic CPU run saves a checkpoint, validation/test metrics and predictions; evaluation of the saved checkpoint returns the same all-day test MSE; changing the input data is rejected when reusing the same output directory.

Additional checks: all Python files parse; both supplied model files match their source snapshots as text; extracted training and evaluation functions match the frozen originals as abstract syntax trees; the inference example and training CLI help run successfully. The release archive excludes raw data, checkpoints, arrays, bytecode, local run directories and machine-specific absolute paths.

On this workstation, the Windows execution sandbox blocked a temporary-file rename during the first end-to-end test. The same tests passed when run with normal process permissions; no training or scientific computation was changed to bypass that restriction.
