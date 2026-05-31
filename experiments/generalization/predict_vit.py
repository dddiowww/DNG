import torch
import os
import sys
import numpy as np
import random
import json
from datetime import datetime
from pathlib import Path
from torch import nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from data.datasets import ViTGraphDataset
from dng_models import ViT_Gen_Predictor
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import argparse
import warnings
warnings.filterwarnings("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

@torch.no_grad()
def eval(model, dl, criterion, device):
    orig_state = model.training
    model.eval()
    pred, actual = [], []
    err, losses = [], []
    with tqdm(total=len(dl)) as pbar:
        for data, acc in dl:
            data = (data[0].to(device), data[1].to(device))
            acc = acc.float().to(device)
            with torch.no_grad():
                pred_acc = model(data).squeeze(-1)
            err.append(torch.abs(pred_acc - acc).mean().item())
            losses.append(criterion(pred_acc, acc).item())
            pred.append(pred_acc.detach().cpu().numpy())
            actual.append(acc.cpu().numpy())
            pbar.update(1)
    avg_err, avg_loss = np.mean(err), np.mean(losses)
    actual, pred = np.concatenate(actual), np.concatenate(pred)
    rsq = r2_score(actual, pred)
    tau = kendalltau(actual, pred).correlation
    model.train(orig_state)
    return avg_err, avg_loss, rsq, tau, actual, pred

def main(args):
    seed = args.seed 
    setup_seed(seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epoch
    
    fourier_dim = args.f_dim
    fourier_scale = args.f_scale
    rnn_mode = args.rnn_mode
    emb_dim = args.n_dim
    head_dim = args.head_dim
    head_drop = args.head_drop

    loss_fn = args.loss_fn
    sigmoid = args.sigmoid

    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'./dng_predict_gen_vit_models/{current_time}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_root = './data/cifar10_vit'
    train_ds = ViTGraphDataset(data_root, 'train')
    val_ds = ViTGraphDataset(data_root, 'val')
    test_ds = ViTGraphDataset(data_root, 'test')
    print('Datasets loaded')
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)

    graph_spec = test_ds.graph_spec

    model = ViT_Gen_Predictor(graph_spec, fourier_dim, fourier_scale, rnn_mode, emb_dim, head_dim, head_drop, sigmoid=sigmoid).to(device)
    print(model)

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: total={total_num:,}, trainable={trainable_num:,}')

    # --- Profile single-sample inference ---
    import time
    model.eval()
    s_data, _ = train_ds[0]
    sample_input = (s_data[0].unsqueeze(0).to(device), s_data[1].unsqueeze(0).to(device))
    with torch.no_grad():
        for _ in range(10):
            model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            torch.cuda.synchronize()
        else:
            peak_mem_mb = 0.0
        t0 = time.time()
        for _ in range(100):
            model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.time() - t0) / 100 * 1000
    try:
        from torch.profiler import profile, ProfilerActivity
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
            with torch.no_grad():
                model(sample_input)
        gflops = sum(e.flops for e in prof.key_averages() if e.flops > 0) / 1e9
    except Exception:
        gflops = float('nan')
    print(f"\nSingle-sample inference profiling:")
    print(f"  Parameters : {total_num:,}")
    print(f"  GFLOPs     : {gflops:.4f}")
    print(f"  Peak memory: {peak_mem_mb:.2f} MB")
    print(f"  Latency    : {latency_ms:.2f} ms (avg over 100 runs)\n")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=args.weight_decay, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(1, 0.5 + 0.0005 * i))
    criterion = {"mse": nn.MSELoss(), "bce": nn.BCELoss()}[loss_fn]
    
    best_rsq, best_tau = -float('inf'), -float('inf')
    best_tau_test, best_tau_epoch = 0.0, -1
    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(total=len(train_dl)) as pbar:
            for data, acc in train_dl:
                data = (data[0].to(device), data[1].to(device))
                acc = acc.float().to(device)
                optimizer.zero_grad()
                pred_acc = model(data).squeeze(-1)
                loss = criterion(pred_acc, acc)
                epoch_loss += loss
                loss.backward()
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                scheduler.step()
                pbar.update(1)
        avg_epoch_loss = epoch_loss/len(train_dl)
        print(avg_epoch_loss)

        # evaluate
        train_avg_err, train_avg_loss, train_rsq, train_tau, _, _ = eval(model, train_dl, criterion, device)
        avg_err, avg_loss, rsq, tau, _, _ = eval(model, val_dl, criterion, device)
        test_avg_err, test_avg_loss, test_rsq, test_tau, _, _ = eval(model, test_dl, criterion, device)
        print(f"Epoch {epoch}, train L1 err: {train_avg_err:.5f}, train loss: {train_avg_loss:.5f}, train Rsq: {train_rsq:.5f}, train tau: {train_tau:.5f}.")
        print(f"Epoch {epoch}, val L1 err: {avg_err:.5f}, val loss: {avg_loss:.5f}, val Rsq: {rsq:.5f}, val tau: {tau:.5f}.")
        print(f"Epoch {epoch}, test L1 err: {test_avg_err:.5f}, test loss: {test_avg_loss:.5f}, test Rsq: {test_rsq:.5f}, test tau: {test_tau:.5f}.")

        save_dict = {
            "weights": model.state_dict(),
            "val_l1": avg_err,
            "val_loss": avg_loss,
            "val_rsq": rsq,
            "epoch": epoch,
        }

        results = {'train_avg_err':train_avg_err, 'train_avg_loss':train_avg_loss, 'train_rsq':train_rsq, 'train_tau':train_tau,
                   'avg_err':avg_err, 'avg_loss':avg_loss, 'rsq':rsq, 'tau':tau,
                   'test_avg_err':test_avg_err, 'test_avg_loss':test_avg_loss, 'test_rsq':test_rsq, 'test_tau':test_tau}

        if rsq > best_rsq:
            val_files = os.listdir(output_dir)
            for f in val_files:
                if 'best_rsq' in f:
                    os.remove(output_dir+f)
            torch.save(save_dict, os.path.join(output_dir, f"best_rsq_{epoch}_{test_rsq}.pt"))
            with open(os.path.join(output_dir, "best_rsq.json"), "w") as f:
                json.dump(results, f, indent=4)
            best_rsq = rsq
        if tau > best_tau:
            val_files = os.listdir(output_dir)
            for f in val_files:
                if 'best_tau' in f:
                    os.remove(output_dir+f)
            torch.save(save_dict, os.path.join(output_dir, f"best_tau_{epoch}_{test_tau}.pt"))
            with open(os.path.join(output_dir, "best_tau.json"), "w") as f:
                json.dump(results, f, indent=4)
            best_tau = tau
            best_tau_test = test_tau
            best_tau_epoch = epoch
            print(f"  ★ New best val τ={tau:.5f} → test τ={test_tau:.5f} (epoch {epoch})")

    print(f"\nDone.")
    print(f"  Best val R²={best_rsq:.5f}")
    print(f"  Best val τ={best_tau:.5f} → test τ={best_tau_test:.5f} (epoch {best_tau_epoch})  ← final result")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNG Predict Generalization — ViT')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--clip-grad', type=float, default=10.0)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--f-scale', type=int, default=1)
    parser.add_argument('--f-dim', type=int, default=32)
    parser.add_argument('--n-dim', type=int, default=32)
    parser.add_argument('--head-dim', type=int, default=256)
    parser.add_argument('--rnn-mode', type=str, default='gru')
    parser.add_argument('--head-drop', type=float, default=0.2)

    parser.add_argument('--loss-fn', type=str, default='bce', choices=['mse', 'bce'])
    parser.add_argument('--sigmoid', action=argparse.BooleanOptionalAction, default=True)
    
    args = parser.parse_args()
    main(args)
