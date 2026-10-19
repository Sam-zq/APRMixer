from types import SimpleNamespace

def config(data_root, station, horizon, seed, workers=0):
    return SimpleNamespace(task_name='long_term_forecast', model='APRMixer', data='pv',
        features='MS', target='OT', freq='15min', root_path=str(data_root),
        data_path='solar_station_%d.csv' % station, station=station, power_capacity={1:50.,2:130.,3:30.,4:130.}[station],
        seq_len=96, label_len=0, pred_len=horizon, enc_in=6, dec_in=6, c_out=1, phys_dim=6,
        d_model=128, n_heads=4, e_layers=2, d_ff=256, dropout=.1, patch_len=16, stride=8,
        persistence_period=96, recent_tau=4., recent_decay=48., recent_mix=.85,
        short_pool_window=96, short_pool_tau=16., short_decay_steps=16.,
        up_residual_scale=.30, down_residual_scale=.60, pv_mark_mode='physics',
        pv_physics_mode='empirical_harmonic', loss='MSE', ramp_loss_weight=0.,
        direction_loss_weight=0., curve_loss_weight=0., event_ramp_weight=2.,
        ramp_event_threshold=.05, correction_loss_weight=0., batch_size=32,
        learning_rate=.0001, train_epochs=10, patience=3, lradj='type1', seed=seed,
        num_workers=workers)
