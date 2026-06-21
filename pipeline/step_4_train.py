"""Stage 4 — model training.

Trains MambaSSM to directly predict log_price_residual at t+1.

Input : 9 single-day snapshots at SLICE_OFFSETS days before anchor t.
Target: residual[t+1]
Loss  : MSE

    Input : config.DM_CSV
    Output: config.CHECKPOINT, config.TRAIN_META_JSON
"""

import json

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from ssm.config import (
    DM_CSV, CHECKPOINT, TRAIN_META_JSON,
    TEST_DAYS, SLICE_OFFSETS, PREDICT_WINDOW,
    EPOCHS, PATIENCE, LR, BATCH_SIZE, VAL_SPLIT, VAL_SEED, WEIGHT_DECAY,
    D_MODEL, N_LAYER, DROPOUT,
    ensure_dirs,
)
from ssm.splits import holdout_cutoff
from ssm.data.loader import create_dataloader
from ssm.arch.mamba import create_model


def build_meta(test_boundary, test_days, best_epoch, best_val_loss):
    return {
        'test_boundary':  test_boundary.strftime('%Y-%m-%d'),
        'test_days':      test_days,
        'slice_offsets':  SLICE_OFFSETS,
        'predict_window': PREDICT_WINDOW,
        'd_model':        D_MODEL,
        'n_layer':        N_LAYER,
        'dropout':        DROPOUT,
        'best_epoch':     best_epoch,
        'best_val_loss':  best_val_loss,
        'run_config': {
            'epochs':       EPOCHS,
            'patience':     PATIENCE,
            'lr':           LR,
            'batch_size':   BATCH_SIZE,
            'val_split':    VAL_SPLIT,
            'weight_decay': WEIGHT_DECAY,
        },
    }


def write_meta(test_boundary, test_days, best_epoch, best_val_loss):
    with open(TRAIN_META_JSON, 'w') as f:
        json.dump(build_meta(test_boundary, test_days, best_epoch, best_val_loss), f, indent=2)


def train():
    ensure_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Task  : predict residual[t+1]  slices={SLICE_OFFSETS}\n')

    full_data  = pd.read_csv(DM_CSV, index_col='Date', parse_dates=True)
    cutoff     = holdout_cutoff(full_data.index)
    train_data = full_data[full_data.index < cutoff]
    test_boundary = full_data.index[len(train_data)]
    test_days     = len(full_data) - len(train_data)

    print(f'Full data       : {full_data.index[0].date()} → {full_data.index[-1].date()}  ({len(full_data)} rows)')
    print(f'Train + val     : {train_data.index[0].date()} → {train_data.index[-1].date()}  ({len(train_data)} rows)')
    print(f'Test (held-out) : {test_boundary.date()} → {full_data.index[-1].date()}  ({test_days} rows)\n')

    write_meta(test_boundary, test_days, best_epoch=0, best_val_loss=float('inf'))

    train_loader, val_loader = create_dataloader(
        train_data, SLICE_OFFSETS, PREDICT_WINDOW, device,
        val_split=VAL_SPLIT, batch_size=BATCH_SIZE, random_seed=VAL_SEED,
    )

    model     = create_model(train_data, PREDICT_WINDOW, device, d_model=D_MODEL, n_layer=N_LAYER, dropout=DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    best_val_loss    = float('inf')
    patience_counter = 0
    best_epoch       = 0

    pbar = tqdm(range(1, EPOCHS + 1), desc='Training', unit='epoch')
    for epoch in pbar:

        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            pred = model(inputs)
            loss = F.mse_loss(pred, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                pred      = model(inputs)
                val_loss += F.mse_loss(pred, targets).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]['lr']

        star = '*' if avg_val < best_val_loss else ' '
        pbar.set_postfix({
            'tr_loss': f'{avg_train:.4f}', 'vl_loss': f'{avg_val:.4f}',
            'lr': f'{lr:.1e}', 'pat': patience_counter, 'best': star,
        })

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT)
            write_meta(test_boundary, test_days, best_epoch, best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                tqdm.write(f'\nEarly stopping at epoch {epoch}.')
                break

    write_meta(test_boundary, test_days, best_epoch, best_val_loss)
    print(f'\nBest val loss : {best_val_loss:.6f}  (epoch {best_epoch})')
    print(f'Checkpoint    → {CHECKPOINT}')


if __name__ == '__main__':
    train()
