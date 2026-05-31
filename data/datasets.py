import os
import torch
import glob
import json
import re
import typing
import random
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, Dataset, ConcatDataset, Subset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from torchvision import transforms
from data.graph_utils import get_siren_shape, get_act_num, get_vit_graph_spec

class SirenGraphDataset(Dataset):
    def __init__(
        self,
        data_path,
        prefix="randinit_test",
        split="all",
        # split point for val and test sets
        split_points: typing.Tuple[int, int] = None,
    ):
        idx_pattern = r"net(\d+)\.pt"
        label_pattern = r"_(\d+)s"
        self.idx_to_path = {}
        self.idx_to_label = {}
        # TODO: this glob pattern should actually be f"{prefix}_[0-9]s/*.pth".
        # For 1 original + 10 augs, this amounts to having 10 copies instead of 11,
        # so it probably doesn't make a big difference in final performance.
        for siren_path in glob.glob(os.path.join(data_path, f"{prefix}_*/*.pt")):
            idx = int(re.search(idx_pattern, siren_path).group(1))
            self.idx_to_path[idx] = siren_path
            label = int(re.search(label_pattern, siren_path).group(1))
            self.idx_to_label[idx] = label
        if split == "all":
            self.idcs = list(range(len(self.idx_to_path)))
        else:
            val_point, test_point = split_points
            self.idcs = {
                "train": list(range(val_point)),
                "val": list(range(val_point, test_point)),
                "test": list(range(test_point, len(self.idx_to_path))),
            }[split]

    def __getitem__(self, idx):
        data_idx = self.idcs[idx]
        graph_data = torch.load(self.idx_to_path[data_idx])
        return graph_data['e'], graph_data['b'], self.idx_to_label[data_idx]

    def __len__(self):
        return len(self.idcs)
    
DEF_TFM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
class INROriginalDataset(Dataset):
    def __init__(
        self, 
        data_type, 
        siren_prefix, 
        split="all", 
        split_points=None, 
        data_tfm=DEF_TFM, 
        img_aug_type='1', 
        noise_level=0.2, 
        train=True
    ):  
        siren_path = f'./data/siren_{data_type}_graph_data'
        data_path ='./data/pytorch_ds'
        self.data_type = data_type
        self.img_aug_type = img_aug_type
        self.noise_level = noise_level
        self.train = train

        sample_file = os.listdir(f'./data/siren_{data_type}_wts/randinit_smaller_0s')[0]
        self.siren_shape = get_siren_shape(os.path.join(f'./data/siren_{data_type}_wts/randinit_smaller_0s', sample_file))
        self.act_num = get_act_num(self.siren_shape, input_layer=True, output_layer=True)

        if img_aug_type != '0':
            if img_aug_type=='1':
                self.h = transforms.RandomHorizontalFlip(p=1)
                self.v = transforms.RandomVerticalFlip(p=1)
                self.r1 = transforms.RandomRotation(degrees=(90, 90))
                self.r2 = transforms.RandomRotation(degrees=(-90, -90))
                self.r3 = transforms.RandomRotation(degrees=(180, 180))
            elif img_aug_type=='2':
                self.norm = transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))

        siren_dset = SirenGraphDataset(siren_path, split="all", prefix=siren_prefix)
        if data_type == "mnist":
            print("Loading MNIST")
            MNIST_train = MNIST(data_path, transform=data_tfm, train=True, download=True)
            MNIST_test = MNIST(data_path, transform=data_tfm, train=False, download=True)
            dset = ConcatDataset([MNIST_train, MNIST_test])
        elif data_type == "fashion":
            print("Loading FashionMNIST")
            fMNIST_train = FashionMNIST(data_path, transform=data_tfm, train=True, download=True)
            fMNIST_test = FashionMNIST(data_path, transform=data_tfm, train=False, download=True)
            dset = ConcatDataset([fMNIST_train, fMNIST_test])
        elif data_type == "cifar":
            print("Loading CIFAR10")
            CIFAR_train = CIFAR10(data_path, transform=data_tfm, train=True, download=True)
            CIFAR_test = CIFAR10(data_path, transform=data_tfm, train=False, download=True)
            dset = ConcatDataset([CIFAR_train, CIFAR_test])
        elif data_type == "cifar100":
            print("Loading CIFAR100")
            CIFAR_train = CIFAR100(data_path, transform=data_tfm, train=True, download=True)
            CIFAR_test = CIFAR100(data_path, transform=data_tfm, train=False, download=True)
            dset = ConcatDataset([CIFAR_train, CIFAR_test])
        
        if split == "all":
            idcs = list(range(len(siren_dset)))
        else:
            val_point, test_point = split_points
            idcs = {
                "train": list(range(val_point)),
                "val": list(range(val_point, test_point)),
                "test": list(range(test_point, len(siren_dset))),
            }[split]
        self.siren_dset = Subset(siren_dset, idcs)
        self.dset = Subset(dset, idcs)

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        img, data_label = self.dset[idx]
        if self.img_aug_type != '0':
            if self.img_aug_type=='1':
                aug_list = [img, self.h(img), self.v(img), self.r1(img), self.r2(img), self.r3(img)]
            elif self.img_aug_type=='2':
                if self.train:
                    noise = torch.randn([3, 32, 32]) if 'cifar' in self.data_type else torch.randn([1, 28, 28])
                    aug_list = [self.norm(torch.clamp(img + noise*self.noise_level, 0, 1))]
                else:
                    aug_list = [self.norm(img)]
            img = torch.cat(aug_list, dim=0) 
        edges, bias, siren_label = self.siren_dset[idx]
        assert siren_label == data_label
        return edges, bias, img, data_label

class ZooGraphDataset(Dataset):
    def __init__(self, data_path, mode, idcs_file=None):
        data = np.load(os.path.join(data_path, "weights.npy"))
        # Hardcoded shuffle order for consistent test set when a split file is provided.
        if idcs_file is not None and os.path.exists(idcs_file):
            shuffled_idcs = pd.read_csv(idcs_file, header=None).values.flatten()
        else:
            shuffled_idcs = np.arange(len(data))
        data = data[shuffled_idcs]
        metrics = pd.read_csv(os.path.join(data_path, "metrics.csv.gz"), compression='gzip')
        metrics = metrics.iloc[shuffled_idcs]
        self.layout = pd.read_csv(os.path.join(data_path, "layout.csv"))
    
        # filter to final-stage weights ("step" == 86 in metrics)
        isfinal = metrics["step"] == 86
        metrics = metrics[isfinal]
        data = data[isfinal]
        assert np.isfinite(data).all()

        metrics.index = np.arange(0, len(metrics))
        idcs = self._split_indices_iid(data)[mode]
        data = data[idcs]
        self.metrics = metrics.iloc[idcs]

        self.metrics.replace({'config.activation': {'relu': 0, 'tanh': 1}}, inplace=True)

        # iterate over rows of layout
        # for each row, get the corresponding weights from data
        self.weights, self.biases, self.act_num, self.layer_type, self.edge_in = [], [], [], [], []
        for i, row in self.layout.iterrows():
            arr = data[:, row["start_idx"]:row["end_idx"]]
            bs = arr.shape[0]
            arr = arr.reshape((bs, *eval(row["shape"])))
            if row["varname"].endswith("kernel:0"):
                self.layer_type.append(row["varname"].split('/')[1])
                # tf to pytorch ordering
                if arr.ndim == 5:
                    arr = torch.tensor(arr).flatten(start_dim=3).flatten(start_dim=1, end_dim=2)
                    if i == 1:
                        self.act_num.append(eval(row["shape"])[-2])
                    self.act_num.append(eval(row["shape"])[-1])
                    self.edge_in.append(arr.shape[1])
                elif arr.ndim == 3:
                    arr = torch.tensor(arr).flatten(start_dim=1)
                    self.act_num.append(eval(row["shape"])[-1])
                    self.edge_in.append(1)
                self.weights.append(arr)
            elif row["varname"].endswith("bias:0"):
                self.biases.append(arr)
            else:
                raise ValueError(f"varname {row['varname']} not recognized.")

    def _split_indices_iid(self, data):
        splits = {}
        test_split_point = int(0.5 * len(data))
        splits["test"] = list(range(test_split_point, len(data)))

        trainval_idcs = list(range(test_split_point))
        val_point = int(0.8 * len(trainval_idcs))
        # use local seed to ensure consistent train/val split
        rng = random.Random(0)
        rng.shuffle(trainval_idcs)
        splits["train"] = trainval_idcs[:val_point]
        splits["val"] = trainval_idcs[val_point:]
        return splits

    def __len__(self):
        return self.weights[0].shape[0]

    def __getitem__(self, idx):
        # insert a channel dim
        edges = tuple(w[idx] for w in self.weights)
        nodes = tuple(b[idx] for b in self.biases)
        return (edges, nodes), self.metrics.iloc[idx].test_accuracy.item(), self.metrics.iloc[idx, 1].item()


class ViTGraphDataset(Dataset):
    def __init__(self, data_root, subset='train'):
        self.data_dir = os.path.join(data_root, 'graph_data')
        label_file = os.path.join(data_root, 'stat/results.json')
        split_file = os.path.join(data_root, 'stat/splits.json')
        hyperparam_file = os.path.join(data_root, 'stat/vit_hyperparameters.json')

        with open(hyperparam_file, 'r') as f:
            hyperparam = json.load(f)
        self.graph_spec = get_vit_graph_spec(hyperparam['depth'], 
                                             hyperparam['dim'], 
                                             hyperparam['dim_head'], 
                                             hyperparam['heads'],
                                             hyperparam['mlp_dim'], 
                                             hyperparam['num_classes'],
                                             hyperparam['patch_size'], 
                                             channels=3)
        
        with open(split_file, 'r') as f:
            splits = json.load(f)
        self.subset_ids = splits[subset]

        with open(label_file, 'r') as f:
            self.labels = json.load(f)

        self.label_map = {item['model_id']: item['test_accuracy']/100 for item in self.labels}
        self.data_files = [f for f in os.listdir(self.data_dir) if int(f.split('_')[1]) in self.subset_ids]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_name = self.data_files[idx]
        data_id = int(file_name.split('_')[1])
        data_path = os.path.join(self.data_dir, file_name)

        data = torch.load(data_path)
        label = self.label_map[data_id]

        return (data['e'], data['b']), label

    
