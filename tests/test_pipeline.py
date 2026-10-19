"""Synthetic diagnostics: they do not establish forecasting accuracy."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
import json

import numpy as np
import pandas as pd
import torch

from configuration import config
from models.APRMixer import Model as CleanModel
from reproduction.APRMixer import Model as ReproductionModel
from data_provider.pv_data_loader import Dataset_PV

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def synthetic_csv(path, days=32):
    dates = pd.date_range('2019-01-01', periods=days*96, freq='15min')
    hours = dates.hour.to_numpy() + dates.minute.to_numpy()/60
    envelope = np.maximum(np.sin(np.pi*(hours-6)/12), 0)
    df = pd.DataFrame({'date': dates, 'Total solar irradiance': envelope*800,
        'Temperature': 20+5*envelope, 'Wind speed': 2+np.cos(hours),
        'Pressure': 1010+np.sin(hours), 'Diffuse irradiance': envelope*120,
        'OT': envelope*35*(.8+.1*np.cos(np.arange(len(dates))/97))})
    df.to_csv(path, index=False)
    return df


class PipelineTests(unittest.TestCase):
    def test_checkpoint_equivalence_and_output_contract(self):
        for horizon in [16, 32, 48, 96]:
            with self.subTest(horizon=horizon):
                c = config(Path('.'), 1, horizon, 2021)
                torch.manual_seed(2021)
                original = ReproductionModel(c).eval()
                clean = CleanModel(c).eval()
                clean.load_state_dict(original.state_dict(), strict=True)
                self.assertEqual(set(clean.state_dict()), set(original.state_dict()))
                self.assertFalse(any(isinstance(n, torch.nn.MultiheadAttention) for n in original.modules()))
                x = torch.randn(2, 96, 6)
                x[..., -1] = torch.rand(2, 96)
                xm = torch.rand(2, 96, 12)
                ym = torch.rand(2, horizon, 12)
                xm[..., -5] = 1
                ym[..., -5] = 1
                ym[:, ::3, -5] = 0
                with torch.no_grad():
                    expected = original(x, xm, None, ym)
                    actual = clean(x, xm, None, ym)
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                self.assertEqual(tuple(actual.shape), (2, horizon, 1))
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(((actual >= 0) & (actual <= 1)).all())
                self.assertEqual(float(actual[:, ::3].abs().max()), 0)

    def test_future_observations_do_not_change_fitted_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'solar_station_1.csv'
            df = synthetic_csv(path)
            c = config(Path(temp), 1, 96, 2021)
            original_train = Dataset_PV(c, temp, flag='train', size=[96, 0, 96], data_path=path.name)
            original_test = Dataset_PV(c, temp, flag='test', size=[96, 0, 96], data_path=path.name)
            boundary = int(len(df)*.7)
            df.loc[boundary:, 'OT'] = 49
            for column in df.columns[1:-1]:
                df.loc[boundary:, column] *= 7
            df.to_csv(path, index=False)
            changed_train = Dataset_PV(c, temp, flag='train', size=[96, 0, 96], data_path=path.name)
            changed_test = Dataset_PV(c, temp, flag='test', size=[96, 0, 96], data_path=path.name)
            np.testing.assert_array_equal(original_train.data_x, changed_train.data_x)
            np.testing.assert_array_equal(original_train.data_stamp, changed_train.data_stamp)
            np.testing.assert_array_equal(original_test.data_stamp, changed_test.data_stamp)
            self.assertFalse(np.array_equal(original_test.data_y, changed_test.data_y))

    def test_train_evaluate_and_run_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            path = folder/'solar_station_1.csv'
            synthetic_csv(path)
            out = folder/'run'
            args = ['--data-root', str(folder), '--station', '1', '--horizon', '16',
                    '--seed', '2021', '--epochs', '1', '--device', 'cpu', '--output', str(out)]
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            def run(script, extra=()):
                return subprocess.run([sys.executable, str(ROOT/script), *args, *extra],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
            trained = run('train.py')
            self.assertEqual(trained.returncode, 0, trained.stdout+trained.stderr)
            before = json.loads((out/'test_metrics.json').read_text())
            evaluated = run('evaluate.py')
            self.assertEqual(evaluated.returncode, 0, evaluated.stdout+evaluated.stderr)
            after = json.loads((out/'test_metrics.json').read_text())
            self.assertEqual(before['all_origins']['all_mse'], after['all_origins']['all_mse'])
            self.assertTrue((out/'TEST_DONE.json').exists())
            changed = pd.read_csv(path)
            changed.loc[0, 'OT'] += .01
            changed.to_csv(path, index=False)
            rejected = run('train.py')
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('different configuration, data or code', rejected.stderr)


if __name__ == '__main__':
    unittest.main()
