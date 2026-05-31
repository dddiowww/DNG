import torch
import os
import sys
import numpy as np
import random
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from torch import nn
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from data.park_dataset import (
    CNNWildParkGraphDataset,
    make_bucket_loaders_with_residual,
)
from dng_models import Gen_Predictor_Park


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def batch_to_device(batch, device):
    return {
        "edge": [e.to(device) for e in batch["edge"]],
        "node": [n.to(device) for n in batch["node"]],
        "mask": [m.to(device) for m in batch["mask"]],
        "layer_node_num": batch["layer_node_num"],
        "labels": batch["labels"].to(device),
        "residual_index": batch["residual_index"].to(device),
        "residual_mask": batch["residual_mask"].to(device),
        "act_ids": batch["act_ids"].to(device),
        "paths": batch["paths"],
        "activations": batch["activations"],
    }


@torch.no_grad()
def evaluate(model, loaders, criterion, device):
    model.eval()
    preds, actuals, losses, errs = [], [], [], []
    for _sig, loader in loaders:
        for batch in loader:
            batch = batch_to_device(batch, device)
            pred = model(batch).squeeze(-1)
            label = batch["labels"]
            errs.append(torch.abs(pred - label).mean().item())
            losses.append(criterion(pred, label).item())
            preds.append(pred.cpu().numpy())
            actuals.append(label.cpu().numpy())
    avg_err = np.mean(errs)
    avg_loss = np.mean(losses)
    actual = np.concatenate(actuals)
    pred = np.concatenate(preds)
    rsq = r2_score(actual, pred)
    tau = kendalltau(actual, pred).correlation
    return avg_err, avg_loss, rsq, tau


def main(args):
    setup_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'./dng_predict_gen_park_models/{current_time}/'
    os.makedirs(output_dir, exist_ok=True)

    # ---- Datasets ----
    print("Loading datasets ...")
    ds_kwargs = dict(
        dataset_dir=args.data_dir,
        splits_path=args.splits_path,
        normalize=args.normalize,
        statistics_path=args.statistics_path,
    )
    train_ds = CNNWildParkGraphDataset(split="train", **ds_kwargs)
    val_ds   = CNNWildParkGraphDataset(split="val",   **ds_kwargs)
    test_ds  = CNNWildParkGraphDataset(split="test",  **ds_kwargs)
    print(f"  train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # ---- Preload all data into RAM ----
    print("Preloading all data into RAM (one-time cost) ...")
    train_ds.preload()
    val_ds.preload()
    test_ds.preload()
    print("  Preload done.")

    # ---- Bucketed DataLoaders ----
    print("Building bucketed loaders (this scans all checkpoints once) ...")
    train_loaders = make_bucket_loaders_with_residual(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loaders = make_bucket_loaders_with_residual(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loaders = make_bucket_loaders_with_residual(
        test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers
    )
    print(f"  train buckets={len(train_loaders)}, val buckets={len(val_loaders)}, test buckets={len(test_loaders)}")

    # ---- Model ----
    model = Gen_Predictor_Park(
        fourier_dim=args.f_dim,
        fourier_scale=args.f_scale,
        rnn_mode=args.rnn_mode,
        rnn_layer=args.rnn_layer,
        emb_dim=args.n_dim,
        att_dim=args.att_dim,
        max_h=args.max_h,
        head_dim=args.head_dim,
        n_input_nodes=3,       # CIFAR-10 → 3 input channels
        n_act_types=6,
        n_classes=10,          # CIFAR-10 → 10 output classes
        head_drop=args.head_drop,
        sigmoid=args.sigmoid,
        drop=args.drop,
    ).to(device)

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: total={total_num:,}, trainable={trainable_num:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(1.0, 0.5 + 0.0005 * i))
    criterion = {"mse": nn.MSELoss(), "bce": nn.BCELoss()}[args.loss_fn]

    # ---- Evaluate before training ----
    print("Evaluating before training ...")
    val_err, val_loss, val_rsq, val_tau = evaluate(model, val_loaders, criterion, device)
    test_err, test_loss, test_rsq, test_tau = evaluate(model, test_loaders, criterion, device)
    print(f"  [init] val  L1={val_err:.5f}  loss={val_loss:.5f}  R²={val_rsq:.5f}  τ={val_tau:.5f}")
    print(f"  [init] test L1={test_err:.5f}  loss={test_loss:.5f}  R²={test_rsq:.5f}  τ={test_tau:.5f}")

    # ---- Training loop ----
    best_rsq, best_tau = -float('inf'), -float('inf')
    best_tau_test, best_tau_epoch = 0.0, -1
    for epoch in range(args.epoch):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        # Shuffle bucket order each epoch
        bucket_order = list(range(len(train_loaders)))
        random.shuffle(bucket_order)

        pbar = tqdm(desc=f"Epoch {epoch}/{args.epoch}", unit="batch")
        for bi in bucket_order:
            _sig, loader = train_loaders[bi]
            for batch in loader:
                batch = batch_to_device(batch, device)
                optimizer.zero_grad()
                pred = model(batch).squeeze(-1)
                loss = criterion(pred, batch["labels"])
                loss.backward()
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                n_batches += 1
                avg_loss = epoch_loss / n_batches
                pbar.set_postfix(loss=f"{loss.item():.5f}", avg=f"{avg_loss:.5f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
                pbar.update(1)
        pbar.close()

        avg_train_loss = epoch_loss / max(n_batches, 1)
        print(f"  train loss: {avg_train_loss:.5f}")

        # ---- Evaluate ----
        val_err, val_loss, val_rsq, val_tau = evaluate(model, val_loaders, criterion, device)
        test_err, test_loss, test_rsq, test_tau = evaluate(model, test_loaders, criterion, device)
        print(f"  val  L1={val_err:.5f}  loss={val_loss:.5f}  R²={val_rsq:.5f}  τ={val_tau:.5f}")
        print(f"  test L1={test_err:.5f}  loss={test_loss:.5f}  R²={test_rsq:.5f}  τ={test_tau:.5f}")

        save_dict = {
            "weights": model.state_dict(),
            "val_l1": val_err, "val_loss": val_loss, "val_rsq": val_rsq, "val_tau": val_tau,
            "test_rsq": test_rsq, "test_tau": test_tau, "epoch": epoch,
        }
        if val_rsq > best_rsq:
            for f in os.listdir(output_dir):
                if 'best_rsq' in f:
                    os.remove(os.path.join(output_dir, f))
            torch.save(save_dict, os.path.join(output_dir, f"best_rsq_{epoch}_{val_rsq:.4f}_{test_rsq:.4f}.pt"))
            best_rsq = val_rsq
        if val_tau > best_tau:
            for f in os.listdir(output_dir):
                if 'best_tau' in f:
                    os.remove(os.path.join(output_dir, f))
            torch.save(save_dict, os.path.join(output_dir, f"best_tau_{epoch}_{val_tau:.4f}_{test_tau:.4f}.pt"))
            best_tau = val_tau
            best_tau_test = test_tau
            best_tau_epoch = epoch
            print(f"  ★ New best val τ={val_tau:.5f} → test τ={test_tau:.5f} (epoch {epoch})")

    print(f"\nDone.")
    print(f"  Best val R²={best_rsq:.5f}")
    print(f"  Best val τ={best_tau:.5f} → test τ={best_tau_test:.5f} (epoch {best_tau_epoch})  ← final result")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNG Predict Generalization — CNN Wild Park')

    parser.add_argument('--data-dir', type=str, default='./data/cnn_park_data',
                        help='Root directory of the CNN park dataset')
    parser.add_argument('--splits-path', type=str, default='cnn_park_splits.json',
                        help='Relative path (from data-dir) to splits JSON')
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--statistics-path', type=str, default='dataset/statistics.pth')

    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--clip-grad', type=float, default=10.0,
                        help='Max gradient norm for clipping (0 = no clip)')
    parser.add_argument('--num-workers', type=int, default=0)

    parser.add_argument('--f-scale', type=int, default=1)
    parser.add_argument('--f-dim', type=int, default=64)
    parser.add_argument('--n-dim', type=int, default=64, help='node embedding dim')
    parser.add_argument('--att-dim', type=int, default=16)
    parser.add_argument('--head-dim', type=int, default=256)
    parser.add_argument('--max-h', type=int, default=49, help='max kernel h*w (7*7=49)')
    parser.add_argument('--rnn-mode', type=str, default='gru')
    parser.add_argument('--rnn-layer', type=int, default=1)
    parser.add_argument('--head-drop', type=float, default=0.2)
    parser.add_argument('--drop', type=float, default=0.2)

    parser.add_argument('--loss-fn', type=str, default='bce', choices=['mse', 'bce'])
    parser.add_argument('--sigmoid', action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    main(args)
