"""Stage 4 — model training.

Loads the model-input frame, holds out the final `TEST_DAYS` as an untouched test
set, builds train/val windows from the remainder, and trains MambaSSM with early
stopping. Persists the best checkpoint and the run metadata stage 5 needs.

    Input : config.MODEL_INPUT_CSV
    Output: config.CHECKPOINT, config.TRAIN_META_JSON
"""

import json

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from ssm.config import (
    MODEL_INPUT_CSV, CHECKPOINT, CHECKPOINT_DIR, TRAIN_META_JSON,
    TEST_DAYS, HISTORIC_HORIZON, FORECAST_HORIZON,
    EPOCHS, PATIENCE, LR, BATCH_SIZE, VAL_SPLIT, D_MODEL, N_LAYER,
    ensure_dirs,
)
from ssm.data.loader import create_dataloader
from ssm.arch.mamba import create_model


# ── main ───────────────────────────────────────────────────────────────────────

def train():
    ensure_dirs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # ── load data, cut off test set ────────────────────────────────────────────

    full_data     = pd.read_csv(MODEL_INPUT_CSV, index_col='Date', parse_dates=True)
    test_boundary = full_data.index[-TEST_DAYS]
    train_data    = full_data.iloc[:-TEST_DAYS]

    print(f'Full data       : {full_data.index[0].date()} → {full_data.index[-1].date()}  ({len(full_data)} rows)')
    print(f'Train + val     : {train_data.index[0].date()} → {train_data.index[-1].date()}  ({len(train_data)} rows)')
    print(f'Test (held-out) : {test_boundary.date()} → {full_data.index[-1].date()}  ({TEST_DAYS} rows)\n')

    train_loader, val_loader = create_dataloader(
        train_data, HISTORIC_HORIZON, FORECAST_HORIZON, device,
        val_split=VAL_SPLIT, batch_size=BATCH_SIZE,
    )

    # ── model, optimiser, scheduler ───────────────────────────────────────────

    model     = create_model(train_data, FORECAST_HORIZON, device, d_model=D_MODEL, n_layer=N_LAYER)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_path = CHECKPOINT

    best_val_loss    = float('inf')
    patience_counter = 0
    best_epoch       = 0

    # ── training loop ─────────────────────────────────────────────────────────

    pbar = tqdm(range(1, EPOCHS + 1), desc='Training', unit='epoch')
    for epoch in pbar:

        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs).squeeze(-1), targets.squeeze(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                val_loss += criterion(model(inputs).squeeze(-1), targets.squeeze(-1)).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        scheduler.step(avg_val)
        lr = optimizer.param_groups[0]['lr']

        star = '*' if avg_val < best_val_loss else ' '
        pbar.set_postfix({'train': f'{avg_train:.5f}', 'val': f'{avg_val:.5f}',
                          'lr': f'{lr:.1e}', 'patience': patience_counter, 'best': star})

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                tqdm.write(f'\nEarly stopping at epoch {epoch} — no val improvement for {PATIENCE} epochs.')
                break

    # ── save metadata for step_5 ──────────────────────────────────────────────

    meta = {
        'test_boundary':    test_boundary.strftime('%Y-%m-%d'),
        'test_days':        TEST_DAYS,
        'historic_horizon': HISTORIC_HORIZON,
        'forecast_horizon': FORECAST_HORIZON,
        'best_epoch':       best_epoch,
        'best_val_loss':    best_val_loss,
    }
    with open(TRAIN_META_JSON, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nBest val loss : {best_val_loss:.6f}  (epoch {best_epoch})')
    print(f'Checkpoint    → {checkpoint_path}')
    print(f'Metadata      → {TRAIN_META_JSON}')


if __name__ == '__main__':
    train()
