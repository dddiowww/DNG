import torch
import os
import sys
import random
from datetime import datetime
from pathlib import Path
import numpy as np
from torch import nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from data.graph_utils import *
from data.datasets import INROriginalDataset
from dng_models import Autoencoder_inr2inr
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from siren_utils import get_batch_siren
from einops import rearrange
import argparse
import warnings
warnings.filterwarnings("ignore") 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def params_to_func_params(weights, biases, bias_id, n_e_connection, act_num):
    func_param = []
    for i in range(len(bias_id)):
        layer_weights = weights[:, n_e_connection[i+1] if i==len(bias_id)-1 else n_e_connection[i+1][0]] # (B, n_l*n_lm1)
        layer_weights = rearrange(layer_weights, 'b (n e) -> b n e', n=act_num[i+1]) # (B, n_l, n_lm1)
        layer_biases = biases[:, bias_id[i]] # (B, n_l)
        func_param.append(layer_weights)
        func_param.append(layer_biases)
    return tuple(func_param)

@torch.no_grad()
def eval_img(model, loader, batch_siren, bias_id, n_e_connection, act_num, device):
    orig_state = model.training
    model.eval()
    recon_loss = 0
    tot_examples = 0
    with tqdm(total=len(loader)) as pbar:
        for e, b, img, y in loader:
            e, b, img = e.to(device), b.to(device), img.to(device)
            data = (e, b)
            with torch.no_grad():
                weights, biases = model(data)
                func_params = params_to_func_params(weights, biases, bias_id, n_e_connection, act_num)
                outs = batch_siren(func_params)
            recon_loss += ((outs - img)**2).mean().item() * img.shape[0]
            tot_examples += img.shape[0]
            pbar.update(1)
    print('loss:', recon_loss / tot_examples)
    model.train(orig_state)
    return recon_loss / tot_examples

@torch.no_grad()
def eval_inr(model, loader, device):
    orig_state = model.training
    model.eval()
    recon_loss = 0
    tot_examples = 0
    with tqdm(total=len(loader)) as pbar:
        for e, b, img, y in loader:
            e, b, img = e.to(device), b.to(device), img.to(device)
            data = (e, b)
            with torch.no_grad():
                weights, biases = model(data)
            loss = ((torch.cat([weights, biases], dim=1) - torch.cat([e, b], dim=1))**2).mean()
            recon_loss += loss.item() * img.shape[0]
            tot_examples += img.shape[0]
            pbar.update(1)
    print('loss:', recon_loss / tot_examples)
    model.train(orig_state)
    return recon_loss / tot_examples

def main(args):
    seed = args.seed 
    setup_seed(seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    ds = args.ds
    aug = args.aug
    extra_aug = 10
    split_points = [55000, 60000] if ds=='fashion' else [45000, 50000]
    batch_size = args.batch_size
    lr = args.lr
    
    fourier_dim = args.f_dim
    fourier_scale = args.f_scale
    rnn_mode = args.rnn_mode
    rnn_layer = args.rnn_layer
    emb_dim = args.n_dim
    latent_dim = args.l_dim
    latent_size = args.l_size
    drop = args.drop

    target = args.target

    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'./dng_encoder_models_inr/{ds}/{current_time}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
    train_ds_noaug = INROriginalDataset(data_type=ds, siren_prefix = 'randinit_smaller', split='train', split_points=split_points, data_tfm=data_tfm, img_aug_type='0')
    val_ds = INROriginalDataset(data_type=ds, siren_prefix = 'randinit_smaller', split='val', split_points=split_points, data_tfm=data_tfm, img_aug_type='0')
    test_ds = INROriginalDataset(data_type=ds, siren_prefix = 'randinit_smaller', split='test', split_points=split_points, data_tfm=data_tfm, img_aug_type='0')

    if aug:
        aug_dsets = []
        for i in range(extra_aug):
            aug_dsets.append(INROriginalDataset(data_type=ds, siren_prefix = f"randinit_smaller_aug{i}", split='train', split_points=split_points, data_tfm=data_tfm, img_aug_type='0'))
        train_ds = ConcatDataset([train_ds_noaug] + aug_dsets)
    print(f"Dataset sizes: train={len(train_ds if aug else train_ds_noaug)}, val={len(val_ds)}, test={len(test_ds)}.")

    train_dl = DataLoader(train_ds if aug else train_ds_noaug, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=100, num_workers=8, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=100, num_workers=8, pin_memory=True)

    siren_shape = test_ds.siren_shape
    act_num = get_act_num(siren_shape, input_layer=True, output_layer=True)
    node_id, _ = get_node_split_idx(act_num)
    edge_id, _ = get_edge_id(node_id)
    layer_bias_id = get_layer_bias_id(act_num)
    n_e_connection_l_s = get_node_edges(edge_id, node_id, layer=True, separate=True)
    n_e_connection = get_node_edges_l_r(n_e_connection_l_s)

    graph_spec = act_num, layer_bias_id, n_e_connection
    weight_num = edge_id[-1][-1][-1] + 1
    bias_num = sum(act_num[1:])

    model = Autoencoder_inr2inr(graph_spec, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim, latent_dim, latent_size, 
                                weight_num, bias_num, drop=drop, ds=ds).to(device)
    print(model)

    encoder_num = sum((p.numel() if 'decoder' not in n else 0) for n, p in model.named_parameters())
    total_num = sum(p.numel() for p in model.parameters())
    print(f'Encoder: {encoder_num}, Total: {total_num}')

    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(1, 0.5 + 0.05 * i))

    if target == 'img':
        batch_siren = get_batch_siren(ds, device)[0]

    best_val_loss = float('inf')

    total_step = 200000
    step = 0
    step_loss = 0
    step_len = 0
    with tqdm(total=total_step) as pbar:
        while step < total_step:
            for e, b, img, _ in train_dl:
                e, b, img = e.to(device), b.to(device), img.to(device)
                data = (e, b)
                optimizer.zero_grad()
                weights, biases = model(data)
                if target == 'img':
                    func_params = params_to_func_params(weights, biases, layer_bias_id, n_e_connection, act_num)
                    outs = batch_siren(func_params)
                    loss = ((outs - img)**2).mean()
                else:
                    loss = ((torch.cat([weights, biases], dim=1) - torch.cat([e, b], dim=1))**2).mean()
                loss.backward()
                optimizer.step()

                step += 1
                pbar.update(1)

                step_loss += loss
                step_len += 1

                if step % 1000 == 0:
                    scheduler.step()
                if step % 2000 == 0:
                    step_avg_loss = step_loss/step_len
                    print('training loss:', step_avg_loss)
                    step_loss = 0
                    step_len = 0

                    print('val set evaluating...')
                    if target == 'img':
                        validation_loss = eval_img(model, val_dl, batch_siren, layer_bias_id, n_e_connection, act_num, device)
                    else:
                        validation_loss = eval_inr(model, val_dl, device)

                    print('testing set evaluating...')
                    if target == 'img':
                        testing_loss = eval_img(model, test_dl, batch_siren, layer_bias_id, n_e_connection, act_num, device)
                    else:
                        testing_loss = eval_inr(model, test_dl, device)
                
                    if validation_loss < best_val_loss:
                        best_val_loss = validation_loss
                        val_files = os.listdir(output_dir)
                        for f in val_files:
                            if 'best_model' in f:
                                os.remove(output_dir+f)
                        torch.save(model.state_dict(), output_dir + f'best_model_{step}.pth')
                        torch.save({
                            "target": target,
                            "step": step,
                            "model": model.state_dict(),
                            "opt": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "testing_loss": testing_loss}, 
                            os.path.join(output_dir, "checkpoint.pt"))

                if step == total_step:
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNG_RE')
    parser.add_argument('--ds', type=str, default='cifar')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    
    parser.add_argument('--aug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--f-scale', type=int, default=3)
    parser.add_argument('--f-dim', type=int, default=128)
    parser.add_argument('--n-dim', type=int, default=512)
    parser.add_argument('--l-dim', type=int, default=256)
    parser.add_argument('--l-size', type=int, default=1)
    parser.add_argument('--rnn-mode', type=str, default='gru')
    parser.add_argument('--rnn-layer', type=int, default=1)
    parser.add_argument('--drop', type=float, default=0.0)

    parser.add_argument('--target', type=str, default='img', choices=['img', 'inr'])
    
    args = parser.parse_args()
    main(args)
