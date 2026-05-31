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
from data.datasets import SirenGraphDataset
from dng_models import DNG_Classifier
from torch.utils.data import DataLoader, ConcatDataset
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
    correct = 0
    tt_len = 0
    tt_loss = 0
    with tqdm(total=len(dl)) as pbar:
        for e, b, label in dl:
            e, b, label = e.to(device), b.to(device), label.to(device)
            data = (e, b)
            with torch.no_grad():
                output = model(data)
            loss = criterion(output, label)
            tt_loss += loss
            correct += torch.sum(torch.eq(output.argmax(-1), label).float())
            tt_len += len(label)
            pbar.update(1)
    acc = correct/tt_len
    avg_loss = tt_loss/len(dl)
    print('acc:', acc)
    print('loss:', avg_loss)
    model.train(orig_state)
    return acc.detach().cpu(), avg_loss.detach().cpu()

def main(args):
    seed = args.seed 
    setup_seed(seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    ds = args.ds
    aug = args.aug
    extra_aug = 20 if ds=='cifar' else 10 
    split_points = [55000, 60000] if ds=='fashion' else [45000, 50000]
    batch_size = args.batch_size
    lr = args.lr
    
    fourier_dim = args.f_dim
    fourier_scale = args.f_scale
    rnn_mode = args.rnn_mode
    rnn_layer = args.rnn_layer
    emb_dim = args.n_dim
    drop = args.drop
    cls_dim = args.cls_dim
    cls_drop = args.cls_drop

    pe = args.pe
    ae = args.ae

    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'./dng_cls_models/{ds}/{current_time}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    siren_path = f'./data/siren_{ds}_graph_data'
    print('loading dataset...')
    train_ds = SirenGraphDataset(siren_path, prefix='randinit_smaller', split="train", split_points=split_points)
    val_ds = SirenGraphDataset(siren_path, prefix='randinit_smaller', split="val", split_points=split_points)
    test_ds = SirenGraphDataset(siren_path, prefix='randinit_smaller', split="test", split_points=split_points)
    
    if aug:
        aug_dsets = []
        for i in range(extra_aug):
            print('loading augmented dataset...')
            aug_dsets.append(SirenGraphDataset(siren_path, prefix = f"randinit_smaller_aug{i}", split="train", split_points=split_points))
        train_ds = ConcatDataset([train_ds] + aug_dsets)
    print(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}.")
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=100, num_workers=8, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=100, num_workers=8, pin_memory=True)
    
    sample_file = os.listdir(f'./data/siren_{ds}_wts/randinit_smaller_0s')[0]
    siren_shape = get_siren_shape(os.path.join(f'./data/siren_{ds}_wts/randinit_smaller_0s', sample_file))
    act_num = get_act_num(siren_shape, input_layer=True, output_layer=True)
    graph_spec = get_graph_spec(act_num)

    model = DNG_Classifier(graph_spec, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim,
                           drop=drop, cls_dim=cls_dim, cls_drop=cls_drop, ds=ds, pe=pe, ae=ae).to(device)
    print(model)

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total: {total_num}, Trainable: {trainable_num}')
  
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(1, 0.5 + 0.05 * i))
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    
    total_step = 300000
    step = 0
    step_loss = 0
    step_len = 0
    with tqdm(total=total_step) as pbar:
        while step < total_step:
            for e, b, label in train_dl:
                e, b, label = e.to(device), b.to(device), label.to(device)
                data = (e, b)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                step += 1
                pbar.update(1)
                
                step_loss += loss
                step_len += 1

                if step % 1000 == 0:
                    scheduler.step()
                    print('lr', optimizer.state_dict()['param_groups'][0]['lr'])

                if step % 2000 == 0:
                    step_avg_loss = step_loss/step_len
                    print('1000 step loss:', step_avg_loss)
                    step_loss = 0
                    step_len = 0

                    print('val set evaluating...')
                    validation_acc, validation_loss = eval(model, val_dl, criterion, device)

                    print('testing set evaluating...')
                    testing_acc, testing_loss= eval(model, test_dl, criterion, device)

                    if validation_acc > best_val_acc:
                        best_val_acc = validation_acc
                        val_files = os.listdir(output_dir)
                        for f in val_files:
                            if 'best_model' in f:
                                os.remove(output_dir+f)
                        torch.save(model.state_dict(), output_dir + f'best_model_{step}.pth')
                        torch.save({
                            "step": step,
                            "model": model.state_dict(),
                            "opt": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict()}, 
                            os.path.join(output_dir, "checkpoint.pt"))
                
                if step == total_step:
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNG_CLASSIFY')
    parser.add_argument('--ds', type=str, default='cifar')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=200, help='number of epoch')
    parser.add_argument('--batch-size', type=int, default=128, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    
    parser.add_argument('--aug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--f-scale', type=int, default=3)
    parser.add_argument('--f-dim', type=int, default=128)
    parser.add_argument('--n-dim', type=int, default=128)
    parser.add_argument('--cls-dim', type=int, default=128)
    parser.add_argument('--rnn-mode', type=str, default='gru')
    parser.add_argument('--rnn-layer', type=int, default=1)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--cls-drop', type=float, default=0.5)

    parser.add_argument('--pe', action=argparse.BooleanOptionalAction)
    parser.add_argument('--ae', action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    main(args)
