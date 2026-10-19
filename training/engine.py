"""Extracted frozen plain-MSE training and evaluation loops."""
from pathlib import Path
import random
import json
import hashlib
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_provider.pv_data_loader import Dataset_PV
from utils.tools import adjust_learning_rate
from evaluation import daylight as metrics
from evaluation import advanced
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def save_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def loader(c, flag):
    ds = Dataset_PV(c, c.root_path, flag=flag, size=[96,0,c.pred_len],
                    features='MS', data_path=c.data_path, target='OT', freq='15min')
    if ds.data_x.shape[-1] != 6:
        raise ValueError('Expected six history variables')
    # Validation is ordered and its loss is aggregated by valid point counts,
    # so a small final batch cannot receive disproportionate weight.
    generator = torch.Generator().manual_seed(c.seed)
    return ds, DataLoader(ds, batch_size=c.batch_size, shuffle=flag=='train',
                         drop_last=False, num_workers=c.num_workers, generator=generator)

def tensors(batch, device):
    return [x.float().to(device) for x in batch]

def call(model, batch, device, horizon):
    x, y, xm, ym = tensors(batch, device)
    return model(x, xm, torch.zeros_like(y), ym), y[:, -horizon:, -1:]

def validation_score(model, dl, device, horizon):
    model.eval()
    sse, sae, count = 0., 0., 0.
    with torch.no_grad():
        for batch in dl:
            p, y = call(model, batch, device, horizon)
            mask = torch.ones_like(y)
            sse += float(((p-y).square()*mask).sum())
            sae += float(((p-y).abs()*mask).sum())
            count += float(mask.sum())
    if count == 0 or not np.isfinite(sse):
        raise ValueError('Invalid validation score')
    return sse/count, sae/count

def train(model, c, out, device, variant):
    _, tr = loader(c, 'train')
    _, va = loader(c, 'val')
    opt = torch.optim.Adam(model.parameters(), lr=c.learning_rate)
    criterion = torch.nn.MSELoss()
    best, bad = float('inf'), 0
    epochs = []
    started = time.perf_counter()
    for epoch in range(c.train_epochs):
        model.train()
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        tick = time.perf_counter()
        losses = []
        for batch in tr:
            opt.zero_grad(set_to_none=True)
            p, y = call(model, batch, device, c.pred_len)
            loss = criterion(p, y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        sync(device)
        seconds = time.perf_counter()-tick
        peak = torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None
        score, val_mae = validation_score(model, va, device, c.pred_len)
        row = dict(epoch=epoch+1, training_seconds=seconds, train_loss=float(np.mean(losses)),
                   val_selection_mse=score, val_selection_mae=val_mae, peak_training_allocated_mib=peak,
                   learning_rate=opt.param_groups[0]['lr'])
        epochs.append(row)
        save_json(out/'epochs.json', epochs)
        print('Epoch %d train=%.7f validation_MAE=%.7f validation_MSE=%.7f time=%.2fs' %
              (epoch+1,row['train_loss'],val_mae,score,seconds), flush=True)
        if score < best:
            best, bad = score, 0
            torch.save(model.state_dict(), out/'checkpoint.tmp')
            (out/'checkpoint.tmp').replace(out/'checkpoint.pth')
            best_epoch = epoch+1
        else:
            bad += 1
        if bad >= c.patience:
            break
        adjust_learning_rate(opt, epoch+1, c)
    summary = dict(best_val_selection_mse=best, selection_metric='all-day global validation MSE',
                   selected_val_mse=epochs[best_epoch-1]['val_selection_mse'],
                   best_epoch=best_epoch, epochs=len(epochs),
                   selected_val_mae=epochs[best_epoch-1]['val_selection_mae'],
                   total_train_and_validation_seconds=time.perf_counter()-started,
                   mean_training_epoch_seconds=float(np.mean([r['training_seconds'] for r in epochs])),
                   peak_training_allocated_mib=max([r['peak_training_allocated_mib'] or 0 for r in epochs])
                       if device.type=='cuda' else None)
    if sys.platform.startswith('linux'):
        import resource
        summary['main_process_peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    save_json(out/'training.json', summary)

def evaluate(model, c, out, device, flag):
    _, dl = loader(c, flag)
    model.eval()
    pp, yy = [], []
    with torch.no_grad():
        for batch in dl:
            p, y = call(model, batch, device, c.pred_len)
            pp.append(p.cpu().numpy())
            yy.append(y.cpu().numpy())
    p, y = np.concatenate(pp), np.concatenate(yy)
    df = pd.read_csv(Path(c.root_path)/c.data_path)
    times = pd.to_datetime(df['date'] if 'date' in df else df.iloc[:,0])
    if times.isna().any() or times.duplicated().any():
        raise ValueError('Invalid evaluation timestamps')
    if not times.is_monotonic_increasing:
        df = df.iloc[np.argsort(times.to_numpy())].reset_index(drop=True)
    n = len(df)
    ntrain, ntest = int(n*.7), int(n*.2)
    begin, end = (ntrain,n-ntest) if flag=='val' else (n-ntest,n)
    if len(p) != end-begin-c.pred_len+1:
        raise ValueError('Evaluation sample count mismatch')
    ix = begin + np.arange(len(p))[:,None] + np.arange(c.pred_len)[None,:]
    power = pd.to_numeric(df.OT, errors='coerce').clip(lower=0)
    power = power.ffill().fillna(power.iloc[:ntrain].median())
    cf = np.clip(power.to_numpy()/c.power_capacity,0,1)
    if np.max(np.abs(cf[ix]-y[...,0])) > 5e-5:
        raise ValueError('Evaluation target alignment failed')
    irr_name = next(k for k in df if 'total solar irradiance' in k.lower())
    irr = pd.to_numeric(df[irr_name],errors='coerce').to_numpy()
    if not np.isfinite(irr[ix]).all():
        raise ValueError('Nonfinite evaluation irradiance')
    day = irr[ix]>1.
    result = {}
    for scope, selection in [('all_origins',slice(None)),('nonoverlap_origins',slice(None,None,c.pred_len))]:
        ps, ys, ds = p[selection,...,0].astype(float), y[selection,...,0].astype(float), day[selection]
        row = metrics.calculate_metrics(ps,ys,ds,.01)
        extra = advanced.calculate_metrics(ps,ys,ds,.01,.05)
        row.update({k:v for k,v in extra.items() if k.startswith('ramp_')})
        result[scope] = row
    save_json(out/(flag+'_metrics.json'),result)
    if flag=='test':
        np.save(out/'pred.npy',p)
        np.save(out/'true.npy',y)
        per_origin=[]
        for origin in range(0,len(p),c.pred_len):
            error=p[origin,:,0].astype(float)-y[origin,:,0].astype(float)
            mask=day[origin]
            per_origin.append(dict(origin=origin,mae=float(np.mean(abs(error))),mse=float(np.mean(error**2)),
                daylight_n=int(mask.sum()),daylight_abs_sum=float(abs(error[mask]).sum()),
                daylight_squared_sum=float((error[mask]**2).sum())))
        metrics.write_csv(out/'per_origin_nonoverlap.csv',per_origin)
