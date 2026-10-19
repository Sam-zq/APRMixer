"""Synthetic tensor-interface example; not a real forecast or accuracy result."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from configuration import config
from models.APRMixer import Model

torch.set_num_threads(1)
c = config(Path('.'), 1, 96, 2021)
model = Model(c).eval()
x = torch.randn(2, 96, 6)
x[..., -1] = torch.rand(2, 96)
past = torch.zeros(2, 96, 12)
future = torch.zeros(2, 96, 12)
for marks in [past, future]:
    marks[..., -6] = .8  # empirical envelope
    marks[..., -5] = 1.  # empirical daylight mask
    marks[..., -4] = 1.  # normalized envelope
    marks[..., -1] = 1.  # duplicated normalized envelope (frozen contract)
with torch.no_grad():
    y = model(x, past, None, future)
print('Synthetic output shape:', tuple(y.shape), 'range:', float(y.min()), float(y.max()))
