"""Tests for graph-construction utilities in data.graph_utils."""

import torch
import pytest
from data.graph_utils import (
    get_act_num,
    get_node_split_idx,
    get_edge_id,
    get_layer_bias_id,
    get_graph_spec,
    get_vit_graph_spec,
    split_head_indices,
)


# ── get_act_num ──────────────────────────────────────────────────────

class TestGetActNum:
    """get_act_num should return per-layer activation counts from weight shapes."""

    @pytest.fixture
    def mlp_shape(self):
        # MLP: 2 → 32 → 1
        weight_spec = [torch.Size([32, 2]), torch.Size([1, 32])]
        bias_spec = [torch.Size([32]), torch.Size([1])]
        return (weight_spec, bias_spec)

    def test_includes_input_and_output(self, mlp_shape):
        act = get_act_num(mlp_shape)
        assert act.tolist() == [2, 32, 1]

    def test_exclude_input(self, mlp_shape):
        act = get_act_num(mlp_shape, input_layer=False)
        assert act.tolist() == [32, 1]

    def test_exclude_output(self, mlp_shape):
        act = get_act_num(mlp_shape, output_layer=False)
        assert act.tolist() == [2, 32]

    def test_deeper_mlp(self):
        # 2 → 16 → 16 → 3
        ws = [torch.Size([16, 2]), torch.Size([16, 16]), torch.Size([3, 16])]
        bs = [torch.Size([16]), torch.Size([16]), torch.Size([3])]
        act = get_act_num((ws, bs))
        assert act.tolist() == [2, 16, 16, 3]


# ── get_node_split_idx ───────────────────────────────────────────────

class TestNodeSplitIdx:
    def test_basic(self):
        node_idx, split_idx = get_node_split_idx(torch.tensor([2, 3, 1]))
        assert node_idx == [[0, 1], [2, 3, 4], [5]]
        assert split_idx == [0, 2, 5, 6]

    def test_single_layer(self):
        node_idx, split_idx = get_node_split_idx(torch.tensor([4]))
        assert node_idx == [[0, 1, 2, 3]]
        assert split_idx == [0, 4]


# ── get_edge_id ──────────────────────────────────────────────────────

class TestEdgeId:
    def test_two_layers(self):
        node_idx = [[0, 1], [2, 3]]
        edge_id, side_nodes = get_edge_id(node_idx)
        assert len(edge_id) == 1
        assert len(side_nodes) == 4  # 2 src * 2 dst

    def test_three_layers(self):
        node_idx = [[0, 1], [2, 3, 4], [5]]
        edge_id, side_nodes = get_edge_id(node_idx)
        assert len(edge_id) == 2
        total_edges = 2 * 3 + 3 * 1  # 6 + 3 = 9
        assert len(side_nodes) == total_edges

    def test_edge_ids_contiguous(self):
        node_idx = [[0, 1], [2, 3, 4], [5]]
        edge_id, _ = get_edge_id(node_idx)
        flat = []
        for layer in edge_id:
            for node_edges in layer:
                flat.extend(node_edges)
        assert flat == list(range(len(flat)))


# ── get_layer_bias_id ────────────────────────────────────────────────

class TestLayerBiasId:
    def test_basic(self):
        bias_id = get_layer_bias_id(torch.tensor([2, 3, 1]))
        assert bias_id == [[0, 1, 2], [3]]

    def test_counts_match(self):
        act_num = torch.tensor([2, 16, 16, 3])
        bias_id = get_layer_bias_id(act_num)
        total = sum(len(layer) for layer in bias_id)
        assert total == sum(act_num[1:]).item()


# ── get_graph_spec ───────────────────────────────────────────────────

class TestGraphSpec:
    def test_returns_three_components(self):
        act_num = torch.tensor([2, 3, 1])
        result = get_graph_spec(act_num)
        assert len(result) == 3

    def test_act_num_preserved(self):
        act_num = torch.tensor([2, 3, 1])
        result_act, _, _ = get_graph_spec(act_num)
        assert result_act.tolist() == [2, 3, 1]


# ── get_vit_graph_spec ──────────────────────────────────────────────

class TestVitGraphSpec:
    @pytest.fixture
    def vit_spec(self):
        return get_vit_graph_spec(
            num_block=2, dim=32, dim_head=16, head=4,
            mlp_dim=64, num_cls=10, patch_size=4, channels=3,
        )

    def test_metadata(self, vit_spec):
        assert vit_spec['num_block'] == 2
        assert vit_spec['dim'] == 32
        assert vit_spec['num_init_node'] == 4 * 4 * 3  # 48

    def test_edge_counts(self, vit_spec):
        assert len(vit_spec['to_emb_edge_id']) == 48 * 32
        assert len(vit_spec['head_edge_id']) == 32 * 10

    def test_node_counts(self, vit_spec):
        assert len(vit_spec['to_emb_node_id']) == 32
        assert len(vit_spec['head_node_id']) == 10

    def test_head_ids(self, vit_spec):
        assert len(vit_spec['head_ids']) == 4  # num heads

    def test_per_block_keys_exist(self, vit_spec):
        for i in range(2):
            for key in ['qkv_edge_id', 'out_edge_id', 'fc1_edge_id',
                        'fc2_edge_id', 'fc1_node_id', 'fc2_node_id']:
                assert f'{key}_{i}' in vit_spec

    def test_edge_ids_non_overlapping(self):
        spec = get_vit_graph_spec(
            num_block=1, dim=8, dim_head=4, head=2,
            mlp_dim=16, num_cls=5, patch_size=2, channels=1,
        )
        seen = set(spec['to_emb_edge_id'])
        for key in ['qkv_edge_id_0', 'out_edge_id_0',
                     'fc1_edge_id_0', 'fc2_edge_id_0', 'head_edge_id']:
            ids = set(spec[key])
            assert seen.isdisjoint(ids), f"{key} overlaps with prior edge ids"
            seen.update(ids)

    def test_edge_ids_contiguous(self):
        spec = get_vit_graph_spec(
            num_block=1, dim=4, dim_head=2, head=2,
            mlp_dim=8, num_cls=3, patch_size=2, channels=1,
        )
        all_ids = spec['to_emb_edge_id'][:]
        all_ids += spec['qkv_edge_id_0']
        all_ids += spec['out_edge_id_0']
        all_ids += spec['fc1_edge_id_0']
        all_ids += spec['fc2_edge_id_0']
        all_ids += spec['head_edge_id']
        assert all_ids == list(range(len(all_ids)))


# ── split_head_indices ───────────────────────────────────────────────

class TestSplitHeadIndices:
    def test_all_indices_present(self):
        result = split_head_indices(list(range(12)), 2)
        assert len(result) == 2
        all_idx = sorted(sum(result, []))
        assert all_idx == list(range(12))

    def test_equal_partition(self):
        result = split_head_indices(list(range(24)), 4)
        for head in result:
            assert len(head) == 6  # 24 / 4

    def test_single_head(self):
        result = split_head_indices(list(range(9)), 1)
        assert len(result) == 1
        assert sorted(result[0]) == list(range(9))
