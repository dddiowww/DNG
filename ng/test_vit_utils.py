"""Tests for NG ViT graph-conversion utilities.

Requires torch_geometric; the entire module is skipped when PyG is absent.
"""

import torch
import pytest

pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from data.graph_utils import get_vit_graph_spec
from ng_vit_utils import get_vit_layer_layout, compute_vit_deg, vit_to_tg_data


# ── get_vit_layer_layout ─────────────────────────────────────────────

class TestGetVitLayerLayout:
    def test_two_blocks(self):
        spec = get_vit_graph_spec(
            num_block=2, dim=32, dim_head=16, head=4,
            mlp_dim=64, num_cls=10, patch_size=4, channels=3,
        )
        layout = get_vit_layer_layout(spec)
        # [n_in, dim] + 4*nb layers + [num_cls]
        assert len(layout) == 2 + 4 * 2 + 1
        assert layout[0] == 48   # 4*4*3
        assert layout[1] == 32   # dim
        assert layout[-1] == 10  # num_cls

    def test_single_block(self):
        spec = get_vit_graph_spec(
            num_block=1, dim=8, dim_head=4, head=2,
            mlp_dim=16, num_cls=5, patch_size=2, channels=1,
        )
        layout = get_vit_layer_layout(spec)
        # [n_in=4, dim=8, inner=8, dim=8, mlp=16, dim=8, cls=5]
        assert layout == [4, 8, 8, 8, 16, 8, 5]


# ── compute_vit_deg ──────────────────────────────────────────────────

class TestComputeVitDeg:
    def test_total_equals_layout_sum(self):
        spec = get_vit_graph_spec(
            num_block=1, dim=8, dim_head=4, head=2,
            mlp_dim=16, num_cls=5, patch_size=2, channels=1,
        )
        deg = compute_vit_deg(spec)
        layout = get_vit_layer_layout(spec)
        assert deg.sum().item() == sum(layout)

    def test_input_nodes_have_degree_zero(self):
        spec = get_vit_graph_spec(
            num_block=1, dim=8, dim_head=4, head=2,
            mlp_dim=16, num_cls=5, patch_size=2, channels=1,
        )
        deg = compute_vit_deg(spec)
        assert deg[0].item() == 4  # n_in = 2*2*1

    def test_dtype(self):
        spec = get_vit_graph_spec(
            num_block=1, dim=4, dim_head=2, head=2,
            mlp_dim=8, num_cls=3, patch_size=2, channels=1,
        )
        deg = compute_vit_deg(spec)
        assert deg.dtype == torch.long


# ── vit_to_tg_data ──────────────────────────────────────────────────

class TestVitToTgData:
    @pytest.fixture
    def tiny_data(self):
        spec = get_vit_graph_spec(
            num_block=1, dim=4, dim_head=2, head=2,
            mlp_dim=8, num_cls=3, patch_size=2, channels=1,
        )
        ne = len(spec['to_emb_edge_id'])
        ne += len(spec['qkv_edge_id_0']) + len(spec['out_edge_id_0'])
        ne += len(spec['fc1_edge_id_0']) + len(spec['fc2_edge_id_0'])
        ne += len(spec['head_edge_id'])

        nn_ = len(spec['to_emb_node_id'])
        nn_ += len(spec['fc1_node_id_0']) + len(spec['fc2_node_id_0'])
        nn_ += len(spec['head_node_id'])

        edges_flat = torch.randn(ne)
        nodes_flat = torch.randn(nn_)
        return spec, edges_flat, nodes_flat

    def test_node_count(self, tiny_data):
        spec, ef, nf = tiny_data
        data = vit_to_tg_data(ef, nf, spec)
        layout = get_vit_layer_layout(spec)
        assert data.x.shape[0] == sum(layout)

    def test_edge_attr_dim(self, tiny_data):
        spec, ef, nf = tiny_data
        data = vit_to_tg_data(ef, nf, spec)
        assert data.edge_attr.shape[1] == 3  # EDGE_DIM

    def test_edge_index_shape(self, tiny_data):
        spec, ef, nf = tiny_data
        data = vit_to_tg_data(ef, nf, spec)
        assert data.edge_index.shape[0] == 2

    def test_target_stored(self, tiny_data):
        spec, ef, nf = tiny_data
        data = vit_to_tg_data(ef, nf, spec, y=0.95)
        assert data.y.item() == pytest.approx(0.95)

    def test_no_target(self, tiny_data):
        spec, ef, nf = tiny_data
        data = vit_to_tg_data(ef, nf, spec, y=None)
        assert not hasattr(data, 'y') or data.y is None
