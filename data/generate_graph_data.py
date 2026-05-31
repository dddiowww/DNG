import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from data.graph_utils import get_siren_shape, get_act_num, get_edge_ft_bias_label

def state_dict_to_tensors(state_dict):
    """Converts a state dict into two lists of equal length:
    1. list of weight tensors
    2. list of biases, or None if no bias
    Assumes the state_dict key order is [0.weight, 0.bias, 1.weight, 1.bias, ...]
    """
    weights, biases = [], []
    keys = list(state_dict.keys())
    i = 0
    while i < len(keys):
        weights.append(state_dict[keys[i]][None])
        i += 1
        assert keys[i].endswith("bias")
        biases.append(state_dict[keys[i]][None])
        i += 1
    return weights, biases

def generate_mlp_graph_data(args):
    '''
    Generate dynamic neural graph data for MLPs or CNNs
    '''
    ds = args.ds
    graph_data_root = f'./data/siren_{ds}_graph_data'

    siren_wts_root = f'./data/siren_{ds}_wts'
    sample_file = os.listdir(os.path.join(siren_wts_root, 'randinit_smaller_0s'))[0]
    print(sample_file)
    siren_shape = get_siren_shape(os.path.join(siren_wts_root, 'randinit_smaller_0s', sample_file))
    act_num = get_act_num(siren_shape, input_layer=True, output_layer=True)

    siren_wts_dir_list = os.listdir(siren_wts_root)
    for i, dir in enumerate(siren_wts_dir_list):
        current_wts_dir = os.path.join(siren_wts_root, dir)
        print(f'{i}/{len(siren_wts_dir_list)} Working on {current_wts_dir}')

        graph_data_dir = os.path.join(graph_data_root, dir)
        if not os.path.exists(graph_data_dir):
            os.makedirs(graph_data_dir)
        
        if len(os.listdir(graph_data_dir)) == len(os.listdir(current_wts_dir)):
            print(f'Already processed {dir}\n')
        else:
            graph_data_list = os.listdir(graph_data_dir)
            graph_origin_len = len(graph_data_list)
            for gf in graph_data_list:
                os.remove(os.path.join(graph_data_dir, gf))
            print(f'Origin Length:{graph_origin_len}. After Remove Length:{len(os.listdir(graph_data_dir))}')
            
            wts_files_list = os.listdir(current_wts_dir)
            with tqdm(total=len(wts_files_list)) as pbar:
                for f in wts_files_list:
                    data_name = f.split('.')[0]
                    sd = torch.load(os.path.join(current_wts_dir, f), map_location='cpu')
                    weights, biases = state_dict_to_tensors(sd)
                    e, b = get_edge_ft_bias_label((weights, biases), act_num, label=False)
                    graph_data = {'e': e, 'b':b}
                    torch.save(graph_data, os.path.join(graph_data_dir, f'{data_name}.pt'))
                    pbar.update(1)
            print(f'Number of processed files:{len(os.listdir(graph_data_dir))}\n')

def generate_vit_graph_data():
    '''
    Generate dynamic neural graph data for CIFAR10 ViTs
    '''
    graph_data_dir = f'./data/cifar10_vit/graph_data'
    if not os.path.exists(graph_data_dir):
        os.makedirs(graph_data_dir)
    vit_models_dir = f'./data/cifar10_vit/vit_models'

    model_list = os.listdir(vit_models_dir)
    for i, m in enumerate(model_list):
        current_wts = os.path.join(vit_models_dir, m)
        print(f'{i}/{len(model_list)} Working on {current_wts}')
        
        data_name = m.split('.')[0]
        model_dict = torch.load(current_wts, map_location='cpu')

        edges = []
        nodes = []
        for k in model_dict:
            if 'to_patch_embedding.2.weight' in k:
                edges.append(model_dict[k].flatten())
            if 'to_patch_embedding.2.bias' in k:
                nodes.append(model_dict[k])
            
            if 'to_qkv.weight' in k:
                edges.append(model_dict[k].flatten())
            if 'to_out.weight' in k:
                edges.append(model_dict[k].flatten())
            if 'net.1.weight' in k:
                edges.append(model_dict[k].flatten())
            if 'net.3.weight' in k:
                edges.append(model_dict[k].flatten())
            if 'net.1.bias' in k:
                nodes.append(model_dict[k])
            if 'net.3.bias' in k:
                nodes.append(model_dict[k])
            
            if 'linear_head.weight' in k:
                edges.append(model_dict[k].flatten())
            if 'linear_head.bias' in k:
                nodes.append(model_dict[k])

        edges = torch.cat(edges)
        nodes = torch.cat(nodes)

        graph_data = {'e': edges, 'b':nodes}
        torch.save(graph_data, os.path.join(graph_data_dir, f'{data_name}.pt'))

    print(f'Number of processed files:{len(os.listdir(graph_data_dir))}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GEN_DATA')
    parser.add_argument('--type', type=str, default='mlp', choices=['mlp', 'vit'])
    parser.add_argument('--ds', type=str, default='cifar')
    
    args = parser.parse_args()
    if args.type == 'mlp':
        generate_mlp_graph_data(args)
    else:
        generate_vit_graph_data()
