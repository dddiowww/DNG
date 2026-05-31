import torch
import os
import sys
import random
import time
import copy
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
from dng_models import Autoencoder, Autoencoder_NonSpatial
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from thop import profile
import argparse
import warnings
warnings.filterwarnings("ignore") 

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def eval(model, dl, criterion, device):
    orig_state = model.training
    model.eval()
    tt_loss = 0
    with tqdm(total=len(dl)) as pbar:
        for e, b, img, y in dl:
            e, b, img = e.to(device), b.to(device), img.to(device)
            data = (e, b)
            with torch.no_grad():
                output = model(data)
            loss = criterion(output, img)
            tt_loss += loss
            pbar.update(1)
    avg_loss = tt_loss/len(dl)
    print('loss:', avg_loss.item())
    model.train(orig_state)
    return avg_loss.detach().cpu()

def eval_time(model, input):
    orig_state = model.training
    model.eval()
    for _ in range(11):
        start = time.time()
        with torch.no_grad():
            output = model(input)
        end = time.time()
        span = end - start
        print(span)
    model.train(orig_state)
    return

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

    nonspatial = args.nonspatial
    
    fourier_dim = args.f_dim
    fourier_scale = args.f_scale
    rnn_mode = args.rnn_mode
    rnn_layer = args.rnn_layer
    emb_dim = args.n_dim
    latent_dim = emb_dim if nonspatial else args.l_dim
    latent_size = args.l_size
    drop = args.drop
    img_aug_type = '0' if nonspatial else args.img_aug_type
    img_aug_size = 6 if img_aug_type == '1' else 1
    noise_level = args.noise_level

    pe = args.pe
    ae = args.ae

    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'./dng_encoder_models_jls/{ds}/{current_time}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if img_aug_type == '2':
        data_tfm = transforms.Compose([transforms.ToTensor()])
    else:
        data_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
    train_ds_noaug = INROriginalDataset(data_type=ds, siren_prefix = 'randinit_smaller', split='train', split_points=split_points, data_tfm=data_tfm, 
                                        img_aug_type=img_aug_type, noise_level=noise_level, train=True)
    val_ds = INROriginalDataset(data_type=ds, siren_prefix = 'randinit_smaller', split='val', split_points=split_points, data_tfm=data_tfm, 
                                img_aug_type=img_aug_type, noise_level=noise_level, train=False)
    test_ds = INROriginalDataset(data_type=ds, siren_prefix = 'randinit_smaller', split='test', split_points=split_points, data_tfm=data_tfm, 
                                 img_aug_type=img_aug_type, noise_level=noise_level, train=False)

    if aug:
        aug_dsets = []
        for i in range(extra_aug):
            aug_dsets.append(INROriginalDataset(data_type=ds, siren_prefix = f"randinit_smaller_aug{i}", split='train', split_points=split_points, data_tfm=data_tfm,
                                                img_aug_type=img_aug_type, noise_level=noise_level, train=True))
        train_ds = ConcatDataset([train_ds_noaug] + aug_dsets)
    print(f"Dataset sizes: train={len(train_ds if aug else train_ds_noaug)}, val={len(val_ds)}, test={len(test_ds)}.")

    train_dl = DataLoader(train_ds if aug else train_ds_noaug, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=100, num_workers=8, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=100, num_workers=8, pin_memory=True)

    siren_shape = test_ds.siren_shape
    act_num = get_act_num(siren_shape, input_layer=True, output_layer=True)
    graph_spec = get_graph_spec(act_num)

    if nonspatial:
        model = Autoencoder_NonSpatial(graph_spec, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim,
                                       drop=drop, ds=ds).to(device)
    else:
        model = Autoencoder(graph_spec, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim, latent_dim, latent_size, 
                            drop=drop, ds=ds, img_aug_size=img_aug_size, pe=pe, ae=ae).to(device)
    print(model)

    # eval inference time
    e_sample, b_sample = test_ds[0][0].unsqueeze(0).to(device), test_ds[0][1].unsqueeze(0).to(device)
    data_sample = (e_sample, b_sample)
    with torch.no_grad():
        output = model(data_sample)
    eval_time(model, data_sample)
    flops, params = profile(copy.deepcopy(model), inputs=(data_sample,))
    print('FLOPs = ' + str(flops*2/1000**3) + 'G')
    
    encoder_num = sum((p.numel() if 'decoder' not in n else 0) for n, p in model.named_parameters())
    total_num = sum(p.numel() for p in model.parameters())
    print(f'Encoder: {encoder_num}, Total: {total_num}')

    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(1, 0.5 + 0.05 * i))
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    total_step = 400000
    step = 0
    step_loss = 0
    step_len = 0
    with tqdm(total=total_step) as pbar:
        while step < total_step:
            for e, b, img, _ in train_dl:
                e, b, img = e.to(device), b.to(device), img.to(device)
                data = (e, b)
                optimizer.zero_grad()
                output = model(data)   
                loss = criterion(output, img)
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
                    validation_loss = eval(model, val_dl, criterion, device)

                    print('testing set evaluating...')
                    testing_loss = eval(model, test_dl, criterion, device)

                    if validation_loss < best_val_loss:
                        best_val_loss = validation_loss
                        val_files = os.listdir(output_dir)
                        for f in val_files:
                            if 'best_model' in f:
                                os.remove(output_dir+f)
                        torch.save(model.state_dict(), output_dir + f'best_model_{step}.pth')
                        torch.save({
                            "step": step,
                            "model": model.state_dict(),
                            "opt": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "best_val_loss": best_val_loss,
                            "testing_loss": testing_loss }, 
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
    parser.add_argument('--l-dim', type=int, default=128)
    parser.add_argument('--l-size', type=int, default=64)
    parser.add_argument('--rnn-mode', type=str, default='gru')
    parser.add_argument('--rnn-layer', type=int, default=1)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--img-aug-type', type=str, default='1', choices=['0', '1', '2'])
    parser.add_argument('--noise-level', type=float, default=0.0)

    parser.add_argument('--nonspatial', action=argparse.BooleanOptionalAction)
    parser.add_argument('--pe', action=argparse.BooleanOptionalAction)
    parser.add_argument('--ae', action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    main(args)
