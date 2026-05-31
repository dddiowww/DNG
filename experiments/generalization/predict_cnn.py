import torch
import os
import sys
import numpy as np
import random
from datetime import datetime
from pathlib import Path
from torch import nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from data.graph_utils import *
from data.datasets import ZooGraphDataset
from dng_models import Gen_Predictor
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
        for data, acc, act in dl:
            data = ([e.to(device) for e in data[0]], [b.to(device) for b in data[1]])
            acc, act = acc.float().to(device), act.to(device)
            with torch.no_grad():
                pred_acc = model(data, act).squeeze(-1)
            err.append(torch.abs(pred_acc - acc).mean().item())
            losses.append(criterion(pred_acc, acc).item())
            pred.append(pred_acc.detach().cpu().numpy())
            actual.append(acc.cpu().numpy())
            pbar.update(1)
    avg_err, avg_loss = np.mean(err), np.mean(losses)
    actual, pred = np.concatenate(actual), np.concatenate(pred)
    rsq = r2_score(actual, pred)
    tau = kendalltau(actual, pred).correlation  # NOTE: on newer scipy this is called "statistic"
    model.train(orig_state)
    return avg_err, avg_loss, rsq, tau, actual, pred

def main(args):
    seed = args.seed 
    setup_seed(seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    ds = args.ds
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epoch
    
    fourier_dim = args.f_dim
    fourier_scale = args.f_scale
    rnn_mode = args.rnn_mode
    rnn_layer = args.rnn_layer
    emb_dim = args.n_dim
    att_dim = args.att_dim
    head_dim = args.head_dim
    head_drop = args.head_drop

    loss_fn = args.loss_fn
    sigmoid = args.sigmoid

    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'./dng_predict_gen_models/{ds}/{current_time}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_path = f'./data/predict_gen/{ds}'
    if ds == 'svhn' and not os.path.exists(data_path):
        data_path = './data/predict_gen/svhn_cropped'
    idcs_file = f'./data/predict_gen/predict_gen_data_splits/{ds}_split.csv'
    train_ds = ZooGraphDataset(data_path=data_path, mode='train', idcs_file=idcs_file)
    val_ds = ZooGraphDataset(data_path=data_path, mode='val', idcs_file=idcs_file)
    test_ds = ZooGraphDataset(data_path=data_path, mode='test', idcs_file=idcs_file)
    print('Datasets loaded')
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=100, num_workers=8, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=100, num_workers=8, pin_memory=True)

    act_num = torch.tensor(train_ds.act_num)
    graph_spec = get_graph_spec(act_num)
    layer_type = train_ds.layer_type
    edge_in = train_ds.edge_in

    model = Gen_Predictor(graph_spec, layer_type, edge_in, fourier_dim, fourier_scale, 
                          rnn_mode, rnn_layer, emb_dim, att_dim, head_dim, head_drop=head_drop, sigmoid=sigmoid).to(device)
    print(model)

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total: {total_num}, Trainable: {trainable_num}')
  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(1, 0.5 + 0.0005 * i))
    criterion = {"mse": nn.MSELoss(), "bce": nn.BCELoss()}[loss_fn]
    
    best_rsq, best_tau = -float('inf'), -float('inf')
    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(total=len(train_dl)) as pbar:
            for data, acc, act in train_dl:
                data = ([e.to(device) for e in data[0]], [b.to(device) for b in data[1]])
                acc, act = acc.float().to(device), act.to(device)
                optimizer.zero_grad()
                pred_acc = model(data, act).squeeze(-1)
                loss = criterion(pred_acc, acc)
                epoch_loss += loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.update(1)
        avg_epoch_loss = epoch_loss/len(train_dl)
        print(avg_epoch_loss)

        # evaluate
        avg_err, avg_loss, rsq, tau, _, _ = eval(model, val_dl, criterion, device)
        test_avg_err, test_avg_loss, test_rsq, test_tau, _, _ = eval(model, test_dl, criterion, device)
        print(f"Epoch {epoch}, val L1 err: {avg_err:.5f}, val loss: {avg_loss:.5f}, val Rsq: {rsq:.5f}, val tau: {tau:.5f}.")
        print(f"Epoch {epoch}, test L1 err: {test_avg_err:.5f}, test loss: {test_avg_loss:.5f}, test Rsq: {test_rsq:.5f}, test tau: {test_tau:.5f}.")

        save_dict = {
            "weights": model.state_dict(),
            "val_l1": avg_err,
            "val_loss": avg_loss,
            "val_rsq": rsq,
            "epoch": epoch,
        }

        if rsq > best_rsq:
            val_files = os.listdir(output_dir)
            for f in val_files:
                if 'best_rsq' in f:
                    os.remove(output_dir+f)
            torch.save(save_dict, os.path.join(output_dir, f"best_rsq_{epoch}_{rsq}_{test_rsq}.pt"))
            best_rsq = rsq
        if tau > best_tau:
            val_files = os.listdir(output_dir)
            for f in val_files:
                if 'best_tau' in f:
                    os.remove(output_dir+f)
            torch.save(save_dict, os.path.join(output_dir, f"best_tau_{epoch}_{tau}_{test_tau}.pt"))
            best_tau = tau

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNGNN_PREDICT_GEN')
    parser.add_argument('--ds', type=str, default='cifar10')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=200, help='number of epoch')
    parser.add_argument('--batch-size', type=int, default=128, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-3)
    
    parser.add_argument('--f-scale', type=int, default=1)
    parser.add_argument('--f-dim', type=int, default=128)
    parser.add_argument('--n-dim', type=int, default=128)
    parser.add_argument('--att-dim', type=int, default=32)
    parser.add_argument('--head-dim', type=int, default=1024)
    parser.add_argument('--rnn-mode', type=str, default='gru')
    parser.add_argument('--rnn-layer', type=int, default=1)
    parser.add_argument('--head-drop', type=float, default=0.0)

    parser.add_argument('--loss-fn', type=str, default='bce')
    parser.add_argument('--sigmoid', action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    main(args)
