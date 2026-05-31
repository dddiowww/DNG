from __future__ import annotations
import os
import sys
import json
import types
import torch

from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from torch.utils.data import Dataset, Subset, DataLoader


class _StubClass:
    """Generic stand-in for any unpickled class from missing packages."""
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

class _StubModule(types.ModuleType):
    """Module that returns _StubClass for any attribute lookup."""
    def __getattr__(self, name):
        fullname = f"{self.__name__}.{name}"
        if fullname in sys.modules:
            return sys.modules[fullname]
        return _StubClass

def _ensure_stub(mod_name):
    parts = mod_name.split(".")
    for i in range(len(parts)):
        sub = ".".join(parts[: i + 1])
        if sub not in sys.modules:
            mod = _StubModule(sub)
            mod.__path__ = []
            mod.__package__ = sub
            sys.modules[sub] = mod

def _safe_torch_load(path, **kwargs):
    """torch.load wrapper that auto-registers stub modules for missing packages."""
    kwargs.setdefault("map_location", "cpu")
    kwargs.setdefault("weights_only", False)
    for _ in range(50):
        try:
            return torch.load(path, **kwargs)
        except ModuleNotFoundError as exc:
            _ensure_stub(str(exc).split("'")[1])
        except AttributeError as exc:
            msg = str(exc)
            if "Can't get attribute" in msg and "on <module" in msg:
                mod_name = msg.split("<module '")[1].split("'")[0]
                _ensure_stub(mod_name)
            else:
                raise
    return torch.load(path, **kwargs)

ACT_NAME_TO_ID = {
    "relu": 0, "gelu": 1, "tanh": 2,
    "sigmoid": 3, "leaky_relu": 4, "none": 5,
}
N_ACT_TYPES = len(ACT_NAME_TO_ID)

class CNNWildParkGraphDataset(Dataset):
    """
    Graph Dataset for CNN models trained on WildPark
    """
    def __init__(
        self,
        dataset_dir: str,
        splits_path: str,
        split: str = "train",
        normalize: bool = False,
        statistics_path: str = "dataset/statistics.pth",
    ):
        self.split = split
        self.dataset_dir = Path(dataset_dir)

        sp = (self.dataset_dir / Path(splits_path)).expanduser().resolve()
        with sp.open("r") as f:
            all_splits = json.load(f)
        if split not in all_splits:
            raise KeyError(f"`split` must be one of {list(all_splits.keys())}, got {split}")
        ds_split = all_splits[split]


        paths = ds_split.get("path", [])
        scores = ds_split.get("score", None)
        if not isinstance(paths, list):
            raise TypeError("splits[split]['path'] must be a list of relative paths.")
        if scores is not None and not isinstance(scores, list):
            raise TypeError("splits[split]['score'] must be a list (or omitted).")
        if scores is not None and len(scores) != len(paths):
            raise ValueError("len(scores) must equal len(path) if provided.")

        self.paths = [(self.dataset_dir / Path(p)).expanduser().resolve() for p in paths]
        self.scores = scores

        self.normalize = normalize
        self.stats = None
        if self.normalize:
            stp = (self.dataset_dir / Path(statistics_path)).expanduser().resolve()
            if not stp.exists():
                raise FileNotFoundError(f"statistics_path not found: {stp}")
            self.stats = _safe_torch_load(stp)
            # expect stats to be in form:
            # {
            #   "weights": {"mean":[Tensor...], "std":[Tensor...]},
            #   "biases":  {"mean":[Tensor...], "std":[Tensor...]},
            # }

    def _normalize(self, weights: List[torch.Tensor], biases: List[torch.Tensor]):
        wm_list, ws_list = self.stats["weights"]["mean"], self.stats["weights"]["std"]
        bm_list, bs_list = self.stats["biases"]["mean"], self.stats["biases"]["std"]

        norm_weights = [
            (w - wm) / ws for w, wm, ws in zip(weights, wm_list, ws_list)
        ]
        norm_biases = [
            (b - bm) / bs for b, bm, bs in zip(biases, bm_list, bs_list)
        ]

        return norm_weights, norm_biases

    @staticmethod
    def _conv_to_multi_edge(W: torch.Tensor) -> torch.Tensor:
        """
        Conv: (out, in, kh, kw) or (B, out, in, kh, kw) -> (B, h=kh*kw, pairs=in*out).

        We consider each input/output channel as a node, and kh*kw weights as the edges between a pair of nodes.
        If we have a layer has out_c = 2, in_c = 3,
        return -> W[:,:, 0]: in_node_0 -> out_node_0
                  W[:,:, 1]: in_node_0 -> out_node_1
                  W[:,:, 2]: in_node_1 -> out_node_0
                  W[:,:, 3]: in_node_1 -> out_node_1
                  W[:,:, 4]: in_node_2 -> out_node_0
                  W[:,:, 5]: in_node_2 -> out_node_1
        6 connections in total, each with kh*kw edges (weight scalars).
        """
        if W.dim() == 4:
            W = W.unsqueeze(0)
        assert W.dim() == 5, f"conv weight must be 4D/5D, got {W.shape}"
        B, out_c, in_c, kh, kw = W.shape
        W = W.permute(0, 2, 1, 3, 4) # (B, out, in, kh, kw) -> (B, in, out, kh, kw)
        W = W.flatten(-2) # (B, in, out, kh*kw)
        W = W.permute(0, 3, 1, 2).reshape(B, kh * kw, in_c * out_c) # (B, kh*kw, in*out)
        return W

    @staticmethod
    def _flatten_linear_to_multi_edge(W: torch.Tensor, prev_out_channels: int) -> torch.Tensor:
        """
        Flatten -> Linear (virtual vertex):
          W: (out, HWC) or (B, out, HWC)
          C = prev_out_channels, h = (HWC // C) = H*W
          -> (B, h, C*out)

        We consider each feature map channel as a input node, each output dimension as a output node,
        E.g., if out_dim=2, prev_out_channels=2, h=4,
        originally we have W: [[w0, w1, w2,  w3,  w4,  w5,  w6,  w7 ],
                               [w8, w9, w10, w11, w12, w13, w14, w15]],
        then we have new W3: [
                                [ w0,  w8,   w4, w12],
                                [ w1,  w9,   w5, w13],
                                [ w2,  w10,  w6, w14],
                                [ w3,  w11,  w7, w15]
                             ],
        where 1st column is in_node_0 -> out_node_0,
              2nd column is in_node_0 -> out_node_1,
              3rd column is in_node_1 -> out_node_0,
              4th column is in_node_1 -> out_node_1.
        """
        if W.dim() == 2:
            W = W.unsqueeze(0)
        B, out_dim, HWC = W.shape
        C = int(prev_out_channels)
        assert HWC % C == 0, f"Linear in-dim {HWC} must be divisible by prev channels C={C}"
        h = HWC // C
        # (B, out, HWC) -> (B, out, C, h) -> (B, h, C*out)
        W4 = W.view(B, out_dim, C, h)
        W3 = W4.permute(0, 3, 2, 1).reshape(B, h, C * out_dim)
        return W3

    @staticmethod
    def _linear_to_multi_edge(W: torch.Tensor) -> torch.Tensor:
        """
        Linear: (out, in) or (B, out, in) -> (B, h=1, pairs=in*out).
        """
        if W.dim() == 2:
            W = W.unsqueeze(0)
        B, out_c, in_c = W.shape
        return W.reshape(B, out_c * in_c).unsqueeze(1)

    @staticmethod
    def _ordered_param_pairs(sd: OrderedDict[str, torch.Tensor]) -> List[Tuple[str, str, str]]:
        """
        E.g. if we have state_dict = OrderedDict([('layers.0.weight', tensor(...)),
                                                  ('layers.0.bias',   tensor(...)),
                                                  ('layers.2.weight', tensor(...)),
                                                  ('layers.2.bias',   tensor(...))])   
        return [('layers.0.weight', 'layers.0.bias', 'layers.0'),
                ('layers.2.weight', 'layers.2.bias', 'layers.2')]
        """
        assert isinstance(sd, OrderedDict), "state_dict should preserve order (OrderedDict)."
        keys = [k for k in sd.keys() if torch.is_tensor(sd[k])]
        used = set()
        pairs: List[Tuple[str, str, str]] = []
        for k in keys:
            if k in used or "weight" not in k:
                continue
            stem = k.rsplit("weight", 1)[0]

            bias_cand = None
            for kk in keys:
                if kk in used:
                    continue
                if "bias" in kk and kk.rsplit("bias", 1)[0] == stem:
                    bias_cand = kk
                    break
            if bias_cand is None:
                raise ValueError(f"Cannot find bias for weight '{k}'")
            
            layer_name = stem[:-1] if stem.endswith('.') else stem
            pairs.append((k, bias_cand, layer_name))
            used.add(k)
            used.add(bias_cand)

        return pairs

    @classmethod
    def _state_to_multi_edge_graph(
        cls,
        state_dict: OrderedDict[str, torch.Tensor],
        expose_layer_names: bool = True
    ) -> Tuple[
        List[torch.Tensor],              # weights_3d:  [(B, edge_num_l, pairs_l)]
        List[torch.Tensor],              # biases_2d:   [(B, out_l)]
        List[int],                       # layer_node_num: [in0, out1, ..., outL]
        List[int],                       # edge_num: [edge_num_l]
        List[str],                       # layer_cls: "conv"/"flatlin"/"linear"
        Optional[List[str]]              # layer_names
    ]:
        weights_3d: List[torch.Tensor] = []
        biases_2d:  List[torch.Tensor] = []
        layer_node_num: List[int] = []
        edge_num: List[int] = []
        layer_cls: List[str] = []
        layer_names: List[str] = []

        pairs = cls._ordered_param_pairs(state_dict)

        prev_out_channels: Optional[int] = None
        prev_was_conv = False
        first_conv_seen = False

        for w_key, b_key, lname in pairs:
            W = state_dict[w_key]
            b = state_dict[b_key]

            if W.dim() == 4:
                # Conv
                out_c, in_c, kh, kw = W.shape
                W3 = cls._conv_to_multi_edge(W)
                if not first_conv_seen:
                    layer_node_num.append(in_c) # input layer node num
                    first_conv_seen = True
                layer_node_num.append(out_c)
                edge_num.append(kh * kw)
                layer_cls.append("conv")
                if expose_layer_names:
                    layer_names.append(lname)

                weights_3d.append(W3)
                biases_2d.append(b.unsqueeze(0))
                prev_out_channels = out_c
                prev_was_conv = True

            elif W.dim() == 2:
                out_c, in_c = W.shape
                # Check if it's Flatten -> Linear
                if prev_was_conv and (prev_out_channels is not None) and (in_c % prev_out_channels == 0):
                    W3 = cls._flatten_linear_to_multi_edge(W, prev_out_channels)
                    h = int(W3.shape[1])
                    layer_node_num.append(out_c)
                    edge_num.append(h)
                    layer_cls.append("flatlin")
                    if expose_layer_names:
                        layer_names.append(lname)
                    weights_3d.append(W3)
                    biases_2d.append(b.unsqueeze(0))
                else:
                    # Linear
                    W3 = cls._linear_to_multi_edge(W)
                    layer_node_num.append(out_c)
                    edge_num.append(1)
                    layer_cls.append("linear")
                    if expose_layer_names:
                        layer_names.append(lname)
                    weights_3d.append(W3)
                    biases_2d.append(b.unsqueeze(0))

                prev_out_channels = out_c
                prev_was_conv = False

            else:
                raise ValueError(f"Unexpected tensor shape for '{w_key}': {W.shape}")

        return (
            weights_3d,
            biases_2d,
            layer_node_num,
            edge_num,
            layer_cls,
            (layer_names if expose_layer_names else None),
        )

    @staticmethod
    def _build_residual_node_pairs(
        residual_cfg: List[int],
        layer_node_num: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Build global residual (u, v) node index pairs.
        """
        if not residual_cfg:
            return []

        offsets = [0]
        for n in layer_node_num:
            offsets.append(offsets[-1] + n)

        pairs: List[Tuple[int, int]] = []
        for dst0, src0 in enumerate(residual_cfg):
            if src0 < 0:
                continue

            src_l = src0 + 1  
            dst_l = dst0 + 1

            c_src = layer_node_num[src_l]
            c_dst = layer_node_num[dst_l]
            m = min(c_src, c_dst)

            base_src = offsets[src_l]
            base_dst = offsets[dst_l]
            for ch in range(m):
                pairs.append((base_src + ch, base_dst + ch))

        return pairs

    @staticmethod
    def _pad_layers_to_edge_max(e_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], int, List[torch.Tensor]]:
        if not e_list:
            return e_list, 0, []

        h_max = max(int(e.shape[1]) for e in e_list)
        if h_max == 0:
            return e_list, 0, []

        padded_edges: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []

        for E in e_list:
            B, h, pairs = E.shape
            if h == h_max:
                padded_edges.append(E)
                mask = torch.ones((B, h_max), dtype=torch.bool, device=E.device)
            else:
                Z = E.new_zeros((B, h_max, pairs))
                Z[:, :h, :] = E
                padded_edges.append(Z)
                mask = torch.zeros((B, h_max), dtype=torch.bool, device=E.device)
                mask[:, :h] = True
            masks.append(mask)

        return padded_edges, h_max, masks

    def __len__(self) -> int:
        return len(self.paths)

    def preload(self):
        """Load and cache all samples into RAM."""
        if hasattr(self, '_cache') and self._cache:
            return
        self._cache: Dict[int, Dict[str, Any]] = {}
        for idx in tqdm(range(len(self)), desc=f"Preloading {self.split}", unit="sample"):
            self._cache[idx] = self._load_item(idx)

    def _load_item(self, idx: int) -> Dict[str, Any]:
        p = self.paths[idx]
        ckpt = _safe_torch_load(p)
        sd = ckpt["model"]

        e_list, n_list, layer_node_num, edge_num_raw, layer_cls, layer_names = \
            self._state_to_multi_edge_graph(sd, expose_layer_names=True)
        
        if self.normalize and self.stats is not None:
            e_list, n_list = self._normalize(e_list, n_list)

        e_list, h_max, masks_list = self._pad_layers_to_edge_max(e_list)

        residual_cfg = ckpt["config"]["residual"]
        activations = ckpt["config"]["activation"]
        residual_pairs = self._build_residual_node_pairs(residual_cfg, layer_node_num)

        act_ids = [ACT_NAME_TO_ID.get(a, ACT_NAME_TO_ID["none"]) for a in activations]
        while len(act_ids) < len(e_list):
            act_ids.append(ACT_NAME_TO_ID["none"])

        label = float(self.scores[idx])

        return {
            "edge": e_list, "node": n_list, "layer_node_num": layer_node_num,
            "edge_num_raw": edge_num_raw, "h_max": h_max, "edge_masks": masks_list,
            "residual_pairs": residual_pairs, "activations": activations,
            "act_ids": act_ids,
            "layer_cls": layer_cls, "layer_names": layer_names or [],
            "num_layers": len(e_list), "path": str(p), "label": label
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if hasattr(self, '_cache') and idx in self._cache:
            return self._cache[idx]
        return self._load_item(idx)


# For batch_size = 1
def collate_as_is(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


# ============== Bucketing by structure ==============
def compute_signature_for_index(ds: CNNWildParkGraphDataset, idx: int) -> Tuple[int, ...]:
    """
    Get layer_node_num tuple for sample at `idx`.
    """
    p = ds.paths[idx]
    ckpt = _safe_torch_load(p)
    sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    if not isinstance(sd, OrderedDict):
        sd = OrderedDict(sd)

    _, _, layer_node_num, _, _, _ = ds._state_to_multi_edge_graph(sd, expose_layer_names=False)
    return tuple(int(x) for x in layer_node_num)


def _iter_weight_keys_in_order(sd: Dict[str, torch.Tensor]):
    if isinstance(sd, OrderedDict):
        keys = [k for k in sd.keys() if ("weight" in k and torch.is_tensor(sd[k]))]
    else:
        keys = sorted([k for k in sd.keys() if ("weight" in k and torch.is_tensor(sd[k]))])
    return keys


def _light_layer_node_num(sd: Dict[str, torch.Tensor]) -> List[int]:
    layer_node_num: List[int] = []
    first_conv_seen = False
    for w_key in _iter_weight_keys_in_order(sd):
        W = sd[w_key]
        if not torch.is_tensor(W):
            continue
        if W.dim() == 4:  # Conv: (out, in, kh, kw)
            out_c, in_c, _, _ = map(int, W.shape)
            if not first_conv_seen:
                layer_node_num.append(in_c)
                first_conv_seen = True
            layer_node_num.append(out_c)
        elif W.dim() == 2:  # Flatten or Linear: (out, in)
            out_c, in_c = map(int, W.shape)
            layer_node_num.append(out_c)
        else:
            continue
    return layer_node_num


def parse_signature_and_residual(ckpt_path: str) -> Tuple[Tuple[int, ...], List[Tuple[int, int]]]:
    """
    Parse checkpoint:
      Returns:
        - signature = tuple(layer_node_num)
        - residual_pairs = List[(src_gid, dst_gid)]
      
      Example:
        residual_cfg = [-1, -1, 0, 1, 2]
        means:
          layer2 <- layer0
          layer3 <- layer1
          layer4 <- layer2
    """
    ckpt = _safe_torch_load(ckpt_path)
    sd = ckpt["model"]

    layer_node_num = _light_layer_node_num(sd)
    residual_cfg = ckpt["config"]["residual"]

    # ---- compute residual pairs ----
    offsets = [0]
    for n in layer_node_num:
        offsets.append(offsets[-1] + n)

    pairs: List[Tuple[int, int]] = []
    for dst_l, src_l in enumerate(residual_cfg):
        if src_l < 0:
            continue
        src_layer = src_l + 1
        dst_layer = dst_l + 1

        c_src = layer_node_num[src_layer]
        c_dst = layer_node_num[dst_layer]
        m = min(c_src, c_dst)

        base_src = offsets[src_layer]
        base_dst = offsets[dst_layer]

        for ch in range(m):
            pairs.append((base_src + ch, base_dst + ch))

    return tuple(layer_node_num), pairs


def build_buckets_with_residual_catalog(ds) -> Tuple[Dict[Tuple[int,...], List[int]], Dict[Tuple[int,...], torch.Tensor]]:
    """
    Build buckets and residual catalog from dataset.
    Returns:
      buckets: dict[signature -> indices]
      residual_catalog: dict[signature -> LongTensor(2, R)], all possible pairs of residual edges for that signature
    """
    buckets = defaultdict(list)
    res_cache = {}

    for i, p in enumerate(tqdm(ds.paths, desc="Bucketing: scan ckpts", unit="ckpt")):
        sig, res_pairs = parse_signature_and_residual(str(p))
        buckets[sig].append(i)
        res_cache[i] = res_pairs  # list[(src,dst)]

    residual_catalog = {}
    for sig, idcs in tqdm(buckets.items(), desc="Build residual catalogs", unit="bucket"):
        all_pairs = []
        for idx in idcs:
            all_pairs.extend(res_cache[idx])
        if not all_pairs:
            residual_catalog[sig] = torch.empty((2, 0), dtype=torch.long)
        else:
            arr = torch.tensor(all_pairs, dtype=torch.long)  # (N,2)
            uniq = torch.unique(arr, dim=0)
            residual_catalog[sig] = uniq.t().contiguous()   # (2,R)

    return dict(buckets), residual_catalog


# ============== Pack a bucket into a batch ==============
def collate_bucket(
    batch: List[Dict[str, Any]],
    residual_index_cat: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Input:
        Samples from the same structural bucket (len = B).  
        Each sample includes (not all shown):
        - "edge": List[L], each layer has shape (1, h_i_l, pairs_l) — already padded to its own h_max within the sample
        - "edge_masks": List[L], each layer (1, h_i_l) — True = valid, False = padded
        - "node": List[L], each layer (1, out_l)
        - "layer_node_num": List[int], includes input layer
        - "label": float
        - "path": str
        - "residual_pairs": List[Tuple[int, int]] — global node index pairs for residual connections
    Optional:
        residual_index_cat: (2, R) — full residual-edge catalog for this bucket  
            (recommended to precompute via `build_buckets_with_residual_catalog`)

    Output:
        Returns a batch dictionary containing:
        - "edge": List[L], (B, H, pairs_l) where H = batch-level max h_max
        - "node": List[L], (B, out_l)
        - "mask": List[L], (B, H)
        - "layer_node_num": Tuple[int, ...]
        - "labels": (B,)
        - "paths": List[str]
        - "residual_index": (2, R)
        - "residual_mask": (B, R) — True if the sample includes that residual edge
    """

    assert len(batch) > 0, "Empty batch."
    B = len(batch)
    L = len(batch[0]["edge"]) # same structure, same L

    # Calculate max h over samples in batch
    H_batch = max(int(x["h_max"]) for x in batch)

    # Check consistency of pairs_l (node pairs) and out_l (nodes) across samples
    pairs_per_layer: List[int] = []
    out_per_layer: List[int] = []
    for l in range(L):
        p_l = [int(x["edge"][l].shape[2]) for x in batch]
        o_l = [int(x["node"][l].shape[1]) for x in batch]
        if len(set(p_l)) != 1:
            raise ValueError(f"pairs mismatch at layer {l}: {set(p_l)}")
        if len(set(o_l)) != 1:
            raise ValueError(f"out dim mismatch at layer {l}: {set(o_l)}")
        pairs_per_layer.append(p_l[0])
        out_per_layer.append(o_l[0])

    # Padding edges and nodes, and generating batch-level masks
    edge_batch: List[torch.Tensor] = []
    node_batch: List[torch.Tensor] = []
    mask_batch: List[torch.Tensor] = []

    for l in range(L):
        pairs_l = pairs_per_layer[l]
        out_l   = out_per_layer[l]

        device = batch[0]["edge"][l].device
        edtype = batch[0]["edge"][l].dtype
        ndtype = batch[0]["node"][l].dtype

        E = torch.zeros((B, H_batch, pairs_l), device=device, dtype=edtype)
        M = torch.zeros((B, H_batch), device=device, dtype=torch.bool)
        N = torch.zeros((B, out_l), device=device, dtype=ndtype)

        for i, sample in enumerate(batch):
            Ei = sample["edge"][l]        # (1, h_i, pairs_l)
            Mi = sample["edge_masks"][l]  # (1, h_i)
            Ni = sample["node"][l]        # (1, out_l)

            _, h_i, _ = Ei.shape
            # pad again to H_batch
            E[i, :h_i, :] = Ei[0]
            M[i, :h_i]    = Mi[0]
            N[i]          = Ni[0]

        edge_batch.append(E)
        mask_batch.append(M)
        node_batch.append(N)

    # Residuals —— catalog and masks
    if residual_index_cat is None:
        residual_index_cat = torch.empty((2, 0), dtype=torch.long)
    R = residual_index_cat.shape[1]
    if R == 0:
        residual_mask = torch.empty((B, 0), dtype=torch.bool, device=residual_index_cat.device)
    else:
        cat_cols = [tuple(residual_index_cat[:, j].tolist()) for j in range(R)]
        rows = []
        for s in batch:
            s_pairs = s["residual_pairs"]
            if not s_pairs:
                rows.append([False] * R)
            else:
                s_set = set(s_pairs)  # O(1) lookup
                rows.append([pair in s_set for pair in cat_cols])
        residual_mask = torch.tensor(rows, dtype=torch.bool, device=residual_index_cat.device)

    # Labels, paths, activations
    labels = torch.tensor([float(x["label"]) for x in batch], dtype=torch.float32)
    paths  = [x["path"] for x in batch]
    layer_node_num = tuple(int(v) for v in batch[0]["layer_node_num"])
    activations = [x["activations"] for x in batch]
    act_ids = torch.tensor([x["act_ids"] for x in batch], dtype=torch.long)  # (B, L)

    return {
        "edge": edge_batch,        # List[L], (B, H, pairs_l)
        "node": node_batch,        # List[L], (B, out_l)
        "mask": mask_batch,        # List[L], (B, H)
        "layer_node_num": layer_node_num,
        "labels": labels,          # (B,)
        "paths": paths,
        "residual_index": residual_index_cat,  # (2, R)
        "residual_mask": residual_mask,        # (B, R)
        "activations": activations, # List[List[str]]
        "act_ids": act_ids,         # (B, L)
    }


# ============== Make bucketed DataLoader ==============
def make_bucket_loaders_with_residual(ds, batch_size=8, shuffle=True, num_workers=0):
    buckets, residual_catalog = build_buckets_with_residual_catalog(ds)

    loaders = []
    for sig, indices in tqdm(buckets.items(), desc="Create loaders", unit="bucket"):
        subset = Subset(ds, indices)
        res_cat = residual_catalog[sig]
        # print(f"Bucket {sig}, size={len(subset)}, residuals={res_cat.shape[1]}")
        def _collate(batch, res_cat=res_cat): 
            return collate_bucket(batch, res_cat)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, collate_fn=_collate, drop_last=False)
        loaders.append((sig, loader))
    return loaders


# ============== Test / Example ==============
if __name__ == "__main__":
    DATASET_ROOT = "./data/cnn_park_data"
    SPLITS_REL   = "cnn_park_splits.json"

    ds = CNNWildParkGraphDataset(
        dataset_dir=DATASET_ROOT,
        splits_path=SPLITS_REL,
        split="test",
        normalize=False,
        statistics_path="dataset/statistics.pth",
    )
    print(f"[INFO] samples in split='{ds.split}': {len(ds)}")

    loaders = make_bucket_loaders_with_residual(
        ds, batch_size=32, shuffle=True, num_workers=0
    )
    print(f"[INFO] total buckets: {len(loaders)}")
    
    done = False
    for sig, loader in loaders:
        for batch in loader:
            # batch:
            # edge: List[L]    -> (B, H, pairs_l)
            # node: List[L]    -> (B, out_l)
            # mask: List[L]    -> (B, H) (True=valid, False=pad)
            # residual_index   -> (2, R)
            # residual_mask    -> (B, R)
            # labels           -> (B,)

            ri = batch["residual_index"]
            rm = batch["residual_mask"]

            if ri.shape[1] > 0:
                print("\n========== Bucket signature ==========")
                print("layer_node_num signature:", sig)

                print("B =", batch["labels"].shape[0])
                print("num_layers =", len(batch["edge"]))
                print("labels:", batch["labels"])

                for l, (E, N, M) in enumerate(zip(batch["edge"], batch["node"], batch["mask"])):
                    print(f"  Layer {l}: edge {tuple(E.shape)}, node {tuple(N.shape)}, mask {tuple(M.shape)}")

                print("residual_index:", ri) # All possible residual edges in this bucket, shape (2, R) means R edges, each defined by (src_node, dst_node)
                print("residual_mask:", rm) # (B, R) True if the sample includes that residual edge
                print("activations:", batch["activations"])
                print("paths[0]:", batch["paths"][0])
                done = True
                break
        
        if done:
            break


    print("\n[OK] data pipeline check passed.")

