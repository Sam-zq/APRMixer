"""Run independent APRMixer training jobs sequentially."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--stations', type=int, nargs='+', choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    p.add_argument('--horizons', type=int, nargs='+', choices=[16, 32, 48, 96], default=[16, 32, 48, 96])
    p.add_argument('--seeds', type=int, nargs='+', default=[2021, 2022, 2023])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=0)
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    for station in a.stations:
        for horizon in a.horizons:
            for seed in a.seeds:
                subprocess.run([sys.executable, str(root/'train.py'), '--data-root', str(a.data_root.resolve()),
                    '--station', str(station), '--horizon', str(horizon), '--seed', str(seed),
                    '--device', a.device, '--workers', str(a.workers)], cwd=root, check=True)


if __name__ == '__main__':
    main()
