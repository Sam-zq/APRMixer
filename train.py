"""Train APRMixer and evaluate its best validation-MSE checkpoint."""
import argparse
import importlib
import inspect
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch

from configuration import config
from training import engine
from data_provider.pv_data_loader import PHYSICS_COLUMNS

ROOT = Path(__file__).resolve().parent


def validate_data(c):
    path = Path(c.root_path) / c.data_path
    df = pd.read_csv(path)
    date_column = 'date' if 'date' in df else df.columns[0]
    times = pd.to_datetime(df[date_column], errors='coerce')
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError('CSV timestamps must be valid, unique and chronologically ordered.')
    if not (times.diff().iloc[1:] == pd.Timedelta(minutes=15)).all():
        raise ValueError('Expected regular 15-minute records; resolve timestamp gaps before training.')
    covariates = [k for k in df if k not in {date_column, 'OT', *PHYSICS_COLUMNS}]
    if len(covariates) != 5:
        raise ValueError('Expected exactly five historical covariates plus OT; got ' + repr(covariates))
    if 'OT' not in df:
        raise ValueError('Missing raw-power target column OT.')
    irradiance = [k for k in covariates if 'total solar irradiance' in k.lower()]
    if not irradiance:
        raise ValueError('Evaluation requires a covariate whose name contains "total solar irradiance".')
    n = len(df)
    ntrain, ntest = int(n * .7), int(n * .2)
    if min(ntrain-c.seq_len, n-ntrain-ntest, ntest) < c.pred_len:
        raise ValueError('CSV is too short for the configured train/validation/test windows.')
    if not np.isfinite(pd.to_numeric(df.OT.iloc[:ntrain], errors='coerce')).any():
        raise ValueError('No finite training power observations.')
    if not np.isfinite(pd.to_numeric(df[irradiance[0]].iloc[ntrain:], errors='coerce')).all():
        raise ValueError('Evaluation irradiance must be finite; its daylight mask uses measured irradiance.')
    return dict(rows=n, columns=list(df.columns), covariates=covariates,
                train_rows=ntrain, validation_rows=n-ntrain-ntest, test_rows=ntest)


def make_signature(c, implementation, dataset_info):
    return dict(config=vars(c), implementation=implementation, dataset=dataset_info,
                data_sha256=engine.sha(Path(c.root_path)/c.data_path),
                source_sha256={p.relative_to(ROOT).as_posix(): engine.sha(p)
                               for p in sorted(ROOT.rglob('*.py'))},
                selection='plain all-day MSE training; global all-day validation MSE')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--station', type=int, choices=[1, 2, 3, 4], default=1)
    p.add_argument('--csv', help='Override solar_station_<station>.csv')
    p.add_argument('--capacity', type=float, help='Installed capacity, in the same units as raw OT')
    p.add_argument('--horizon', type=int, choices=[16, 32, 48, 96], default=96)
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--implementation', choices=['reproduction', 'clean'], default='reproduction')
    p.add_argument('--output', type=Path)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--evaluate-only', action='store_true', help='Re-evaluate an existing matching run')
    return p.parse_args()


def main():
    a = parse_args()
    if a.epochs < 1 or a.workers < 0:
        raise ValueError('epochs must be positive and workers nonnegative.')
    device = torch.device(a.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. For a diagnostic run, explicitly use --device cpu.')
        torch.cuda.set_device(device)
    c = config(a.data_root.resolve(), a.station, a.horizon, a.seed, a.workers)
    if a.csv:
        c.data_path = a.csv
    if a.capacity is not None:
        if not np.isfinite(a.capacity) or a.capacity <= 0:
            raise ValueError('capacity must be finite and positive.')
        c.power_capacity = a.capacity
    c.train_epochs = a.epochs
    info = validate_data(c)
    out = (a.output or ROOT/'runs'/f'APRMixer_{a.implementation}_s{a.station}_h{a.horizon}_seed{a.seed}').resolve()
    out.mkdir(parents=True, exist_ok=True)
    signature = make_signature(c, a.implementation, info)
    sig_path = out/'config.json'
    if sig_path.exists():
        previous = json.loads(sig_path.read_text(encoding='utf-8'))
        if previous != signature:
            raise ValueError('Existing run has different configuration, data or code. Choose another --output.')
    elif a.evaluate_only:
        raise FileNotFoundError('Evaluation requires config.json and checkpoint.pth from a matching run.')
    else:
        engine.save_json(sig_path, signature)
    module = 'reproduction.APRMixer' if a.implementation == 'reproduction' else 'models.APRMixer'
    engine.seed_all(c.seed)
    model = importlib.import_module(module).Model(c).float().to(device)
    engine.seed_all(c.seed + 10000)
    if not a.evaluate_only:
        if (out/'TEST_DONE.json').exists():
            print('Completed run:', out)
            return
        engine.save_json(out/'environment.json', dict(python=platform.python_version(),
            torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
            cuda=torch.version.cuda, device=str(device),
            gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            deterministic_cudnn=True, amp=False))
        if not (out/'TRAIN_DONE.json').exists():
            engine.train(model, c, out, device, 'APRMixer')
            engine.save_json(out/'TRAIN_DONE.json', dict(complete=True))
    load_options = {'weights_only': True} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    checkpoint = torch.load(out/'checkpoint.pth', map_location=device, **load_options)
    model.load_state_dict(checkpoint, strict=True)
    engine.evaluate(model, c, out, device, 'val')
    engine.evaluate(model, c, out, device, 'test')
    engine.save_json(out/'TEST_DONE.json', dict(complete=True, checkpoint_sha256=engine.sha(out/'checkpoint.pth')))
    print('Saved best checkpoint, metrics and predictions:', out)


if __name__ == '__main__':
    main()
