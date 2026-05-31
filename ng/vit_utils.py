import torch
from torch_geometric.data import Data

# Edge feature dimension: 3 for QKV edges, padded to 3 for others
EDGE_DIM = 3


def get_vit_layer_layout(graph_spec):
    """Compute the layer layout (nodes per layer) for a ViT.

    Returns:
        list[int]: [n_input, dim, inner, dim, mlp, dim, ..., num_cls]
    """
    nb = graph_spec['num_block']
    dim = graph_spec['dim']
    inner = graph_spec['dim_head'] * graph_spec['head']
    mlp = graph_spec['mlp_dim']
    nc = graph_spec['num_cls']
    n_in = graph_spec['num_init_node']

    layout = [n_in, dim]
    for _ in range(nb):
        layout.extend([inner, dim, mlp, dim])
    layout.append(nc)
    return layout


def compute_vit_deg(graph_spec):
    """Compute in-degree histogram for PNA from ViT graph structure."""
    nb = graph_spec['num_block']
    dim = graph_spec['dim']
    inner = graph_spec['dim_head'] * graph_spec['head']
    mlp = graph_spec['mlp_dim']
    nc = graph_spec['num_cls']
    n_in = graph_spec['num_init_node']

    max_deg = max(n_in, dim, inner, mlp) + 1
    deg = torch.zeros(max_deg, dtype=torch.long)

    deg[0] += n_in
    deg[n_in] += dim

    for _ in range(nb):
        deg[dim] += inner
        deg[inner] += dim
        deg[dim] += mlp
        deg[mlp] += dim

    deg[dim] += nc

    return deg


def vit_to_tg_data(edges_flat, nodes_flat, graph_spec, y=None):
    """Convert DNG-format flattened ViT weights/biases to a PyG Data object.

    Following the NG paper's transformer graph construction:
      - QKV edges have 3-D features (W_Q_ij, W_K_ij, W_V_ij)
      - All other edges have 1-D features zero-padded to 3-D

    Args:
        edges_flat: 1-D tensor of all concatenated flattened weight matrices.
        nodes_flat: 1-D tensor of all concatenated bias vectors.
        graph_spec: dict from get_vit_graph_spec().
        y: optional scalar target (e.g. test accuracy).

    Returns:
        torch_geometric.data.Data with x, edge_index, edge_attr (E, 3), layer_layout.
    """
    nb = graph_spec['num_block']
    dim = graph_spec['dim']
    inner = graph_spec['dim_head'] * graph_spec['head']
    mlp = graph_spec['mlp_dim']
    nc = graph_spec['num_cls']
    n_in = graph_spec['num_init_node']

    layout = get_vit_layer_layout(graph_spec)
    total_nodes = sum(layout)

    # Cumulative node offsets per layer
    cum = [0]
    for s in layout:
        cum.append(cum[-1] + s)

    x = torch.zeros(total_nodes, 1)

    x[cum[1]:cum[1] + dim, 0] = nodes_flat[graph_spec['to_emb_node_id']]

    for i in range(nb):
        base = 2 + 4 * i  # layer index for QKV-head nodes of block i
        # QKV head nodes, Out nodes: no bias → stays zero
        # FC1 bias → layer base+2
        x[cum[base + 2]:cum[base + 2] + mlp, 0] = nodes_flat[graph_spec[f'fc1_node_id_{i}']]
        # FC2 bias → layer base+3
        x[cum[base + 3]:cum[base + 3] + dim, 0] = nodes_flat[graph_spec[f'fc2_node_id_{i}']]

    # Head bias → last hidden layer → output
    x[cum[-2]:cum[-2] + nc, 0] = nodes_flat[graph_spec['head_node_id']]

    all_src, all_dst, all_attr = [], [], []

    def _add_fc_1d(src_start, n_src, dst_start, n_dst, w_flat, out_feat, in_feat):
        """Add fully-connected edges with 1-D features (padded to 3-D).

        PyTorch weight: (out_feat, in_feat). w[j, i] = edge from i→j.
        """
        w = w_flat.reshape(out_feat, in_feat).T  # (in, out) = (n_src, n_dst)
        src = torch.arange(n_src, dtype=torch.long) + src_start
        dst = torch.arange(n_dst, dtype=torch.long) + dst_start
        sg, dg = torch.meshgrid(src, dst, indexing='ij')
        all_src.append(sg.reshape(-1))
        all_dst.append(dg.reshape(-1))
        # Pad scalar weight to 3-D: [w, 0, 0]
        attr = torch.zeros(n_src * n_dst, EDGE_DIM)
        attr[:, 0] = w.reshape(-1)
        all_attr.append(attr)

    def _add_qkv(src_start, n_src, dst_start, n_dst, qkv_flat, dim_in):
        """Add QKV edges with 3-D features (W_Q_ij, W_K_ij, W_V_ij).

        qkv_flat: flattened weight of Linear(dim_in, 3*inner).
        PyTorch weight shape: (3*inner, dim_in).
        Split into Q (inner, dim), K (inner, dim), V (inner, dim).
        """
        w_qkv = qkv_flat.reshape(3 * n_dst, dim_in)  # (3*inner, dim)
        w_q = w_qkv[:n_dst, :]       # (inner, dim)
        w_k = w_qkv[n_dst:2*n_dst, :]
        w_v = w_qkv[2*n_dst:, :]

        wq_t = w_q.T  # (dim, inner)
        wk_t = w_k.T
        wv_t = w_v.T

        src = torch.arange(n_src, dtype=torch.long) + src_start
        dst = torch.arange(n_dst, dtype=torch.long) + dst_start
        sg, dg = torch.meshgrid(src, dst, indexing='ij')
        all_src.append(sg.reshape(-1))
        all_dst.append(dg.reshape(-1))

        attr = torch.stack([
            wq_t.reshape(-1),
            wk_t.reshape(-1),
            wv_t.reshape(-1),
        ], dim=-1)  # (n_src * n_dst, 3)
        all_attr.append(attr)

    _add_fc_1d(cum[0], n_in, cum[1], dim,
               edges_flat[graph_spec['to_emb_edge_id']], dim, n_in)

    for i in range(nb):
        base = 2 + 4 * i
        prev_layer = base - 1  # previous layer has dim nodes

        # QKV: dim → inner, 3-D edge features (Q, K, V)
        _add_qkv(cum[prev_layer], dim, cum[base], inner,
                 edges_flat[graph_spec[f'qkv_edge_id_{i}']], dim)

        # Out projection: Linear(inner, dim) — 1-D edge features
        _add_fc_1d(cum[base], inner, cum[base + 1], dim,
                   edges_flat[graph_spec[f'out_edge_id_{i}']], dim, inner)

        # FC1: Linear(dim, mlp) — 1-D edge features
        _add_fc_1d(cum[base + 1], dim, cum[base + 2], mlp,
                   edges_flat[graph_spec[f'fc1_edge_id_{i}']], mlp, dim)

        # FC2: Linear(mlp, dim) — 1-D edge features
        _add_fc_1d(cum[base + 2], mlp, cum[base + 3], dim,
                   edges_flat[graph_spec[f'fc2_edge_id_{i}']], dim, mlp)

    _add_fc_1d(cum[-3], dim, cum[-2], nc,
               edges_flat[graph_spec['head_edge_id']], nc, dim)

    edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)])  # (2, E)
    edge_attr = torch.cat(all_attr, dim=0)                              # (E, 3)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        layer_layout=layout,
    )
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)

    return data
