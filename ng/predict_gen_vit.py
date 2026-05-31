import torch
import os
import sys
import numpy as np
import random
import json
import math
import argparse
import warnings
from datetime import datetime
from torch import nn
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    os.path.join(_script_dir, '..', 'neural-graphs-main'),   # local: Dynamic-Neural-Graph/../
    os.path.join(_script_dir, 'neural-graphs-main'),          # server: DNG/neural-graphs-main
    os.path.join(_script_dir, '..', '..', 'neural-graphs-main'),  # fallback
]
NG_DIR = next((os.path.abspath(p) for p in _candidates if os.path.isdir(p)), _candidates[0])
sys.path.insert(0, NG_DIR)

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj

from data.datasets import ViTGraphDataset
from ng_vit_utils import vit_to_tg_data, get_vit_layer_layout, compute_vit_deg, EDGE_DIM


class GaussianEncoding(nn.Module):
    """Random Fourier Features: x → [sin(Bx), cos(Bx)]."""

    def __init__(self, sigma: float, input_size: int, encoded_size: int):
        super().__init__()
        self.register_buffer('B', torch.randn(input_size, encoded_size) * sigma)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class NG_GNN_ViT(nn.Module):
    """NG-GNN baseline for ViT generalization prediction.

    Pipeline (all sparse):
      scalar bias  →  GaussianEncoding(1→d) → Linear → + pos_embed → PNA → pool → MLP → scalar
      3-D weight   →  GaussianEncoding(3→d) → Linear →                PNA edge
    """

    def __init__(
        self,
        d_hid: int,
        layer_layout: list,
        n_input_nodes: int,
        num_classes: int,
        deg: torch.Tensor,
        sin_emb_dim: int = 128,
        inp_factor: float = 1.0,
        n_gnn_layers: int = 4,
        pooling_method: str = 'mean',
        pooling_layer_idx: str = 'last',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_layout = layer_layout
        self.num_classes = num_classes
        self.n_input_nodes = n_input_nodes

        self.node_enc = nn.Sequential(
            GaussianEncoding(sigma=inp_factor, input_size=1, encoded_size=sin_emb_dim),
            nn.Linear(2 * sin_emb_dim, d_hid),
        )
        self.edge_enc = nn.Sequential(
            GaussianEncoding(sigma=inp_factor, input_size=EDGE_DIM, encoded_size=sin_emb_dim),
            nn.Linear(2 * sin_emb_dim, d_hid),
        )

        num_hidden_layers = len(layer_layout) - 2
        total_pos = n_input_nodes + num_hidden_layers + num_classes
        self.pos_embed = nn.Parameter(torch.randn(total_pos, d_hid))

        pos_idx = []
        for i in range(layer_layout[0]):
            pos_idx.append(i)
        for k in range(num_hidden_layers):
            idx = n_input_nodes + k
            pos_idx.extend([idx] * layer_layout[k + 1])
        for i in range(num_classes):
            pos_idx.append(n_input_nodes + num_hidden_layers + i)
        self.register_buffer('pos_idx', torch.tensor(pos_idx, dtype=torch.long))

        from nn.gnn import PNA
        self.gnn = PNA(
            in_channels=d_hid,
            hidden_channels=d_hid,
            num_layers=n_gnn_layers,
            out_channels=d_hid,
            aggregators=['mean', 'min', 'max', 'std'],
            scalers=['identity', 'amplification'],
            edge_dim=d_hid,
            deg=deg,
            dropout=dropout,
            norm='layernorm',
            act='silu',
            update_edge_attr=True,
            modulate_edges=True,
        )

        from nn.pooling import HeterogeneousAggregator
        self.pool = HeterogeneousAggregator(
            d_hid, d_hid, d_hid,
            pooling_method, pooling_layer_idx,
            n_input_nodes, num_classes,
        )

        num_graph_features = d_hid
        if pooling_method == 'cat' and pooling_layer_idx == 'last':
            num_graph_features = num_classes * d_hid

        self.proj_out = nn.Sequential(
            nn.Linear(num_graph_features, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, 1),
        )

    def forward(self, batch):
        x = self.node_enc(batch.x)            # (N_total, d_hid)
        ea = self.edge_enc(batch.edge_attr)    # (E_total, d_hid)

        batch_size = batch.num_graphs
        pos = self.pos_embed[self.pos_idx]     # (nodes_per_sample, d_hid)
        pos_batch = pos.repeat(batch_size, 1)  # (N_total, d_hid)
        x = x + pos_batch

        out_node, _out_edge = self.gnn(x=x, edge_index=batch.edge_index, edge_attr=ea)

        dense_node, node_mask = to_dense_batch(out_node, batch.batch)
        layer_layouts = [self.layer_layout] * batch_size
        graph_feat = self.pool(dense_node, layer_layouts, node_mask=node_mask)

        return self.proj_out(graph_feat)


class NG_T_ViT(nn.Module):
    """NG-T baseline for ViT generalization prediction.

    Pipeline:
      Fourier-encode features (1-D bias, 3-D edge) → dense → RTLayer × n_layers → pool → MLP
    Note: requires O(N²) memory for dense edge features where N = total neurons.
    """

    def __init__(
        self,
        d_node: int,
        d_edge: int,
        d_attn_hid: int,
        d_node_hid: int,
        d_edge_hid: int,
        d_out_hid: int,
        layer_layout: list,
        n_input_nodes: int,
        num_classes: int,
        sin_emb_dim: int = 128,
        inp_factor: float = 1.0,
        n_layers: int = 4,
        n_heads: int = 4,
        pooling_method: str = 'mean',
        pooling_layer_idx: str = 'last',
        dropout: float = 0.0,
        modulate_v: bool = True,
    ):
        super().__init__()
        self.layer_layout = layer_layout
        self.num_classes = num_classes
        self.n_input_nodes = n_input_nodes
        self.total_nodes = sum(layer_layout)

        self.node_enc = nn.Sequential(
            GaussianEncoding(sigma=inp_factor, input_size=1, encoded_size=sin_emb_dim),
            nn.Linear(2 * sin_emb_dim, d_node),
        )
        self.edge_enc = nn.Sequential(
            GaussianEncoding(sigma=inp_factor, input_size=EDGE_DIM, encoded_size=sin_emb_dim),
            nn.Linear(2 * sin_emb_dim, d_edge),
        )

        num_hidden_layers = len(layer_layout) - 2
        total_pos = n_input_nodes + num_hidden_layers + num_classes
        self.pos_embed = nn.Parameter(torch.randn(total_pos, d_node))

        pos_idx = []
        for i in range(layer_layout[0]):
            pos_idx.append(i)
        for k in range(num_hidden_layers):
            idx = n_input_nodes + k
            pos_idx.extend([idx] * layer_layout[k + 1])
        for i in range(num_classes):
            pos_idx.append(n_input_nodes + num_hidden_layers + i)
        self.register_buffer('pos_idx', torch.tensor(pos_idx, dtype=torch.long))

        from nn.relational_transformer import RTLayer
        self.layers = nn.ModuleList([
            RTLayer(
                d_node, d_edge, d_attn_hid, d_node_hid, d_edge_hid,
                n_heads, float(dropout),
                node_update_type='rt',
                disable_edge_updates=(i == n_layers - 1),
                modulate_v=modulate_v,
                use_ln=True,
                tfixit_init=False,
                n_layers=n_layers,
            )
            for i in range(n_layers)
        ])

        from nn.pooling import HeterogeneousAggregator
        self.pool = HeterogeneousAggregator(
            d_node, d_out_hid, d_node,
            pooling_method, pooling_layer_idx,
            n_input_nodes, num_classes,
        )

        num_graph_features = d_node
        if pooling_method == 'cat' and pooling_layer_idx == 'last':
            num_graph_features = num_classes * d_node

        self.proj_out = nn.Sequential(
            nn.Linear(num_graph_features, d_out_hid),
            nn.ReLU(),
            nn.Linear(d_out_hid, d_out_hid),
            nn.ReLU(),
            nn.Linear(d_out_hid, 1),
        )

    def forward(self, batch):
        x = self.node_enc(batch.x)            # (N_total, d_node)
        ea = self.edge_enc(batch.edge_attr)    # (E_total, d_edge)

        batch_size = batch.num_graphs
        pos = self.pos_embed[self.pos_idx]
        pos_batch = pos.repeat(batch_size, 1)
        x = x + pos_batch

        dense_node, node_mask = to_dense_batch(x, batch.batch)           # (B, N, d_node)
        dense_edge = to_dense_adj(batch.edge_index, batch.batch, ea)     # (B, N, N, d_edge)
        mask = to_dense_adj(batch.edge_index, batch.batch).unsqueeze(-1) # (B, N, N, 1)

        node_feat = dense_node
        edge_feat = dense_edge
        for layer in self.layers:
            node_feat, edge_feat = layer(node_feat, edge_feat, mask)

        layer_layouts = [self.layer_layout] * batch_size
        graph_feat = self.pool(node_feat, layer_layouts, node_mask=node_mask)

        return self.proj_out(graph_feat)


class ViTGraphDatasetPyG(torch.utils.data.Dataset):
    """Wraps ViTGraphDataset to return PyG Data objects."""

    def __init__(self, vit_ds: ViTGraphDataset, graph_spec: dict):
        self.vit_ds = vit_ds
        self.graph_spec = graph_spec

    def __len__(self):
        return len(self.vit_ds)

    def __getitem__(self, idx):
        (edges, nodes), label = self.vit_ds[idx]
        return vit_to_tg_data(edges, nodes, self.graph_spec, y=label)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_sigmoid=True):
    model.eval()
    preds, actuals, losses, errs = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).squeeze(-1)
        if use_sigmoid:
            pred = torch.sigmoid(pred)
        label = batch.y.squeeze(-1)
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
    output_dir = f'./ng_predict_gen_vit_models/{args.model}/{current_time}/'
    os.makedirs(output_dir, exist_ok=True)

    data_root = args.data_root
    print("Loading datasets ...")
    train_ds_raw = ViTGraphDataset(data_root, 'train')
    val_ds_raw = ViTGraphDataset(data_root, 'val')
    test_ds_raw = ViTGraphDataset(data_root, 'test')
    graph_spec = train_ds_raw.graph_spec
    print(f"  train={len(train_ds_raw)}, val={len(val_ds_raw)}, test={len(test_ds_raw)}")

    train_ds = ViTGraphDatasetPyG(train_ds_raw, graph_spec)
    val_ds = ViTGraphDatasetPyG(val_ds_raw, graph_spec)
    test_ds = ViTGraphDatasetPyG(test_ds_raw, graph_spec)

    train_dl = PyGDataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = PyGDataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_dl = PyGDataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    layout = get_vit_layer_layout(graph_spec)
    n_in = graph_spec['num_init_node']
    nc = graph_spec.get('num_cls', 10)
    total_nodes = sum(layout)
    print(f"  ViT graph: {len(layout)} layers, {total_nodes} nodes, layout={layout}")

    if args.model == 'ng-gnn':
        deg = compute_vit_deg(graph_spec).to(device)
        model = NG_GNN_ViT(
            d_hid=args.d_hid,
            layer_layout=layout,
            n_input_nodes=n_in,
            num_classes=nc,
            deg=deg,
            sin_emb_dim=args.sin_emb_dim,
            inp_factor=args.inp_factor,
            n_gnn_layers=args.n_layers,
            pooling_method=args.pooling_method,
            pooling_layer_idx=args.pooling_layer_idx,
            dropout=args.dropout,
        ).to(device)
    elif args.model == 'ng-t':
        mem_est = total_nodes ** 2 * args.d_edge * 4 * args.batch_size / 1e9
        print(f"  NG-T dense memory estimate: ~{mem_est:.1f} GB for edge features")
        if mem_est > 16:
            print("  WARNING: Dense edge features may exceed GPU memory. "
                  "Consider reducing --batch-size or --d-edge.")
        model = NG_T_ViT(
            d_node=args.d_hid,
            d_edge=args.d_edge,
            d_attn_hid=args.d_attn_hid,
            d_node_hid=args.d_node_hid,
            d_edge_hid=args.d_edge_hid,
            d_out_hid=args.d_out_hid,
            layer_layout=layout,
            n_input_nodes=n_in,
            num_classes=nc,
            sin_emb_dim=args.sin_emb_dim,
            inp_factor=args.inp_factor,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            pooling_method=args.pooling_method,
            pooling_layer_idx=args.pooling_layer_idx,
            dropout=args.dropout,
            modulate_v=args.modulate_v,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model [{args.model}]: total={total_params:,}, trainable={trainable_params:,}")

    # --- Profile single-sample inference ---
    import time
    model.eval()
    sample_batch = Batch.from_data_list([train_ds[0]]).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(sample_batch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        model(sample_batch)
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            model(sample_batch)
        torch.cuda.synchronize()
        latency_ms = (time.time() - t0) / 100 * 1000
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                      record_shapes=True, with_flops=True) as prof:
            with torch.no_grad():
                model(sample_batch)
        gflops = sum(e.flops for e in prof.key_averages() if e.flops > 0) / 1e9
    except Exception:
        gflops = float('nan')
    print(f"\n  Single-sample inference profiling:")
    print(f"    Parameters : {total_params:,}")
    print(f"    GFLOPs     : {gflops:.4f}")
    print(f"    Peak memory: {peak_mem_mb:.2f} MB")
    print(f"    Latency    : {latency_ms:.2f} ms (avg over 100 runs)\n")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: min(1.0, 0.5 + 0.0005 * i)
    )
    criterion = {"mse": nn.MSELoss(), "bce": nn.BCELoss()}[args.loss_fn]

    use_sigmoid = args.sigmoid
    best_rsq, best_tau = -float('inf'), -float('inf')
    best_tau_test, best_tau_epoch = 0.0, -1
    for epoch in range(args.epoch):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}", unit="batch")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch).squeeze(-1)
            if use_sigmoid:
                pred = torch.sigmoid(pred)
            label = batch.y.squeeze(-1)
            loss = criterion(pred, label)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.5f}")
        pbar.close()
        print(f"  train loss: {epoch_loss / max(n_batches, 1):.5f}")

        val_err, val_loss, val_rsq, val_tau = evaluate(model, val_dl, criterion, device, use_sigmoid)
        test_err, test_loss, test_rsq, test_tau = evaluate(model, test_dl, criterion, device, use_sigmoid)
        print(f"  val  L1={val_err:.5f}  loss={val_loss:.5f}  R²={val_rsq:.5f}  τ={val_tau:.5f}")
        print(f"  test L1={test_err:.5f}  loss={test_loss:.5f}  R²={test_rsq:.5f}  τ={test_tau:.5f}")

        save_dict = {
            "weights": model.state_dict(),
            "val_l1": val_err, "val_loss": val_loss, "val_rsq": val_rsq, "val_tau": val_tau,
            "test_rsq": test_rsq, "test_tau": test_tau, "epoch": epoch,
            "args": vars(args),
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
    parser = argparse.ArgumentParser(description='NG-GNN / NG-T baselines for ViT generalization prediction')

    parser.add_argument('--model', type=str, default='ng-gnn', choices=['ng-gnn', 'ng-t'])

    parser.add_argument('--data-root', type=str, default='./data/cifar10_vit')

    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--clip-grad', type=float, default=10.0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--loss-fn', type=str, default='bce', choices=['mse', 'bce'])
    parser.add_argument('--sigmoid', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--d-hid', type=int, default=56, help='Node hidden dim (d_node for NG-T)')
    parser.add_argument('--sin-emb-dim', type=int, default=128, help='Fourier encoding dim')
    parser.add_argument('--inp-factor', type=float, default=1.0, help='Fourier scale σ')
    parser.add_argument('--n-layers', type=int, default=4, help='GNN / Transformer layers')
    parser.add_argument('--pooling-method', type=str, default='mean',
                        choices=['mean', 'max', 'cat'])
    parser.add_argument('--pooling-layer-idx', type=str, default='last',
                        choices=['last', 'all'])

    parser.add_argument('--d-edge', type=int, default=56)
    parser.add_argument('--d-attn-hid', type=int, default=56)
    parser.add_argument('--d-node-hid', type=int, default=56)
    parser.add_argument('--d-edge-hid', type=int, default=56)
    parser.add_argument('--d-out-hid', type=int, default=56)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--modulate-v', action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    main(args)
