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
from einops import rearrange 
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from data.graph_utils import *
from data.datasets import INROriginalDataset
from dng_models import Autoencoder, Autoencoder_inr2inr, Autoencoder_NonSpatial
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

class ConvLatentClassifier(nn.Module):
    def __init__(self, ds='cifar', in_dim=256, drop=0.2, cls_num=10):
        super().__init__()
        self.ds = ds
        if 'cifar' in ds:
            self.conv = nn.Sequential(nn.Conv2d(in_dim, 256, 3, stride=1, padding=1),  # 8*8
                                      nn.BatchNorm2d(num_features=256),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2), # 4*4
                                      nn.Dropout(drop),

                                      nn.Conv2d(256, 256, 3, stride=1, padding=1), # 4*4
                                      nn.BatchNorm2d(num_features=256),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2), # 2*2
                                      nn.Dropout(drop),

                                      nn.Conv2d(256, 256, 1, stride=1), # 2*2
                                      nn.BatchNorm2d(num_features=256),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2), # 1*1
                                      )
            self.head = nn.Sequential(nn.Linear(256, 256),             
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(256, 256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(256, cls_num))
   
        else: 
            self.conv = nn.Sequential(nn.Conv2d(in_dim, 256, 3, stride=2, padding=1),  # 4*4
                                      nn.BatchNorm2d(num_features=256),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2), # 2*2
                                      nn.Dropout(drop),

                                      nn.Conv2d(256, 256, 1, stride=1), # 2*2
                                      nn.BatchNorm2d(num_features=256),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2), # 1*1
                                      )
            self.head = nn.Sequential(nn.Linear(256, 256),             
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(256, 256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(256, cls_num))
        
    def forward(self, x):
        if 'cifar' in self.ds:
            x = rearrange(x, 'b (v h w) c -> (b v) c h w ', h=8, w=8)
        else:
            x = rearrange(x, 'b (v h w) c -> (b v) c h w ', h=7, w=7)
        x = self.conv(x).flatten(start_dim=1)
        x = self.head(x)
        return x

class MLPLatentClassifier(nn.Module):
    def __init__(self, mode, input_dim, hidden_dim, ds):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(input_dim * (3 if 'cifar' in ds and mode=='jls' else 1), hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, 100 if '100' in ds else 10))
    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.head(x)

class Sample_Model(nn.Module):
    def __init__(self, graph_spec, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim, latent_dim, latent_size, 
                 drop=0.0, ds='cifar', img_aug_size=6, cls_drop=0.2):
        super().__init__()
        self.encoder = Autoencoder(graph_spec, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim, latent_dim, latent_size, 
                                   drop=drop, ds=ds, img_aug_size=img_aug_size)
        self.classifier = ConvLatentClassifier(ds=ds, in_dim=latent_dim, drop=cls_drop, cls_num=(100 if ds=='cifar100' else 10))
    def forward(self, x):
        x = self.encoder.encode(x)
        x = self.classifier(x)
        return x

@torch.no_grad()
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

@torch.no_grad()
def eval(encoder, classifier, dl, device, img_aug_size=6):
    orig_state = classifier.training
    classifier.eval()
    correct = 0
    tt_len = 0
    with tqdm(total=len(dl)) as pbar:
        for e, b, _, label in dl:
            e, b, label = e.to(device), b.to(device), label.to(device)
            data = (e, b)
            with torch.no_grad():
                latent = encoder.encode(data)
                output = classifier(latent)
            output = rearrange(output, '(b v) d -> b v d', v=img_aug_size)
            output = torch.sum(output, dim=1)
            correct += torch.sum(torch.eq(output.argmax(-1), label).float())
            tt_len += len(label)
            pbar.update(1)
    acc = correct/tt_len
    print('acc:', acc)
    classifier.train(orig_state)
    return acc.detach().cpu()

def main(args):
    seed = args.seed 
    re_seed = args.re_seed 
    setup_seed(seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    ds = args.ds
    aug = args.aug
    extra_aug = 10
    split_points = [55000, 60000] if ds=='fashion' else [45000, 50000]
    batch_size = args.batch_size
    lr = args.lr

    mode = args.mode
    inr_target = args.inr_target
    jls_nonspatial = args.jls_nonspatial
    
    fourier_dim = args.f_dim
    fourier_scale = args.f_scale
    rnn_mode = args.rnn_mode
    rnn_layer = args.rnn_layer
    emb_dim = args.n_dim
    latent_dim = emb_dim if (mode=='jls' and jls_nonspatial) else args.l_dim
    latent_size = args.l_size
    drop = args.drop
    cls_drop = args.cls_drop
    img_aug_type = '0' if (mode=='inr' or jls_nonspatial) else args.img_aug_type
    img_aug_size = 6 if img_aug_type == '1' else 1
    noise_level = args.noise_level

    pe = args.pe
    ae = args.ae

    enc_dir = args.enc_dir

    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'./dng_latent_cls_models_{mode}/{ds}/{current_time}/'
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
    node_id, _ = get_node_split_idx(act_num)
    edge_id, _ = get_edge_id(node_id)
    layer_bias_id = get_layer_bias_id(act_num)
    n_e_connection_l_s = get_node_edges(edge_id, node_id, layer=True, separate=True)
    n_e_connection = get_node_edges_l_r(n_e_connection_l_s)

    graph_spec = act_num, layer_bias_id, n_e_connection

    # eval model info
    if mode=='jls' and not jls_nonspatial:
        e_sample, b_sample = test_ds[0][0].unsqueeze(0).to(device), test_ds[0][1].unsqueeze(0).to(device)
        data_sample = (e_sample, b_sample)
        model = Sample_Model(graph_spec, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim, latent_dim, latent_size, 
                             drop=drop, ds=ds, img_aug_size=img_aug_size, cls_drop=cls_drop).to(device)
        eval_time(model, data_sample)
        flops, params = profile(copy.deepcopy(model), inputs=(data_sample,))
        print('FLOPs = ' + str(flops*2/1000**3) + 'G')

    if mode == 'jls':
        if jls_nonspatial:
            encoder = Autoencoder_NonSpatial(graph_spec, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim,
                                             drop=drop, ds=ds).to(device)
        else:
            encoder = Autoencoder(graph_spec, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim, latent_dim, latent_size, 
                                  drop=drop, ds=ds, img_aug_size=img_aug_size, pe=pe, ae=ae).to(device)
        encoder_path = enc_dir
    elif mode == 'inr':
        weight_num = edge_id[-1][-1][-1] + 1
        bias_num = sum(act_num[1:])
        encoder = Autoencoder_inr2inr(graph_spec, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim, latent_dim, latent_size, 
                                      weight_num, bias_num, drop=drop, ds=ds).to(device)
        encoder_path = enc_dir
    files = os.listdir(encoder_path)
    for f in files:
        if 'best_model' in f:
            file = f
    print('Encoder: ', file)
    encoder.load_state_dict(torch.load(os.path.join(encoder_path, file)), strict=True)
    encoder.eval()
    if mode == 'jls':
        if jls_nonspatial:
            classifier = MLPLatentClassifier(mode, emb_dim, 256, ds).to(device)
        else:
            classifier = ConvLatentClassifier(ds=ds, in_dim=latent_dim, drop=cls_drop, cls_num=(100 if ds=='cifar100' else 10)).to(device)
    elif mode == 'inr':
        classifier = MLPLatentClassifier(mode, latent_dim, 256, ds).to(device)
    print(classifier)

    enc_num = sum((p.numel() if 'decoder' not in n else 0) for n, p in encoder.named_parameters())
    cls_num = sum(p.numel() for p in classifier.parameters())
    total_num = enc_num + cls_num
    print(f'Encoder: {enc_num}, Classifier: {cls_num}, Total: {total_num}')

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(1, 0.3 + 0.07 * i))
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    
    total_step = 200000
    step = 0
    step_loss = 0
    step_len = 0
    with tqdm(total=total_step) as pbar:
        while step < total_step:
            for e, b, _, label in train_dl:
                e, b, label = e.to(device), b.to(device), label.to(device)
                data = (e, b)
                label = label.repeat_interleave(img_aug_size)

                optimizer.zero_grad()
                with torch.no_grad():
                    latent = encoder.encode(data)
                output = classifier(latent)
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

                    step_avg_loss = step_loss/step_len
                    print('1000 step loss:', step_avg_loss)
                    step_loss = 0
                    step_len = 0

                    print('val set evaluating...')
                    validation_acc = eval(encoder, classifier, val_dl, device, img_aug_size)

                    print('testing set evaluating...')
                    testing_acc = eval(encoder, classifier, test_dl, device, img_aug_size)
            
                    if validation_acc > best_val_acc:
                        best_val_acc = validation_acc
                        val_files = os.listdir(output_dir)
                        for f in val_files:
                            if 'best_model' in f:
                                os.remove(output_dir+f)
                        torch.save(classifier.state_dict(), output_dir + f'best_model_{step}_{testing_acc}.pth')
                        torch.save({
                            "step": step,
                            "encoder": encoder.state_dict(),
                            "classifier": classifier.state_dict(),
                            "opt": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "best_val_acc": best_val_acc,
                            "testing_acc": testing_acc }, 
                            os.path.join(output_dir, "checkpoint.pt"))
                    
                if step == total_step:
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNG_LATENT_CLS')
    parser.add_argument('--ds', type=str, default='cifar')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--re-seed', type=int, default=0)
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
    parser.add_argument('--cls-drop', type=float, default=0.2)
    parser.add_argument('--img-aug-type', type=str, default='1', choices=['0', '1', '2'])
    parser.add_argument('--noise-level', type=float, default=0.0)

    parser.add_argument('--mode', type=str, default='jls', choices=['jls', 'inr'])
    parser.add_argument('--inr-target', type=str, default='img', choices=['img', 'inr'])
    parser.add_argument('--jls-nonspatial', action=argparse.BooleanOptionalAction)
    parser.add_argument('--pe', action=argparse.BooleanOptionalAction)
    parser.add_argument('--ae', action=argparse.BooleanOptionalAction)

    parser.add_argument('--enc-dir', type=str, default='dng_encoder_models')
    
    args = parser.parse_args()
    main(args)
