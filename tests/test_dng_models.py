"""Tests for DNG model forward passes and shape correctness."""

import torch
import pytest
from data.graph_utils import get_graph_spec, get_vit_graph_spec
from dng_models import (
    GaussianFourierFeatureTransform,
    GRUCell,
    RNNCell,
    DynamicNeuralGNN,
    DynamicNeuralGNN_Trans,
    ViT_Gen_Predictor,
)


# ── helpers ──────────────────────────────────────────────────────────

def _count_edges_nodes_for_vit(spec):
    """Return (num_edges, num_nodes) from a ViT graph spec."""
    ne = len(spec['to_emb_edge_id'])
    nn_ = len(spec['to_emb_node_id'])
    for i in range(spec['num_block']):
        ne += len(spec[f'qkv_edge_id_{i}'])
        ne += len(spec[f'out_edge_id_{i}'])
        ne += len(spec[f'fc1_edge_id_{i}'])
        ne += len(spec[f'fc2_edge_id_{i}'])
        nn_ += len(spec[f'fc1_node_id_{i}'])
        nn_ += len(spec[f'fc2_node_id_{i}'])
    ne += len(spec['head_edge_id'])
    nn_ += len(spec['head_node_id'])
    return ne, nn_


# ── GaussianFourierFeatureTransform ──────────────────────────────────

class TestGaussianFourier:
    def test_output_shape(self):
        layer = GaussianFourierFeatureTransform(in_channels=1, mapping_size=32, scale=10)
        x = torch.randn(2, 100)
        out = layer(x)
        assert out.shape == (2, 64, 100)  # mapping_size * 2

    def test_deterministic(self):
        layer = GaussianFourierFeatureTransform(mapping_size=16)
        x = torch.randn(1, 50)
        assert torch.allclose(layer(x), layer(x))


# ── GRUCell / RNNCell ────────────────────────────────────────────────

class TestGRUCell:
    def test_output_shape(self):
        cell = GRUCell(32, 32)
        x = torch.randn(4, 10, 32)
        h = torch.randn(4, 10, 32)
        assert cell(x, h).shape == (4, 10, 32)

    def test_different_dims(self):
        cell = GRUCell(16, 32)
        x = torch.randn(2, 5, 16)
        h = torch.randn(2, 5, 32)
        assert cell(x, h).shape == (2, 5, 32)


class TestRNNCell:
    def test_output_shape(self):
        cell = RNNCell(32, 32)
        x = torch.randn(4, 10, 32)
        h = torch.randn(4, 10, 32)
        assert cell(x, h).shape == (4, 10, 32)


# ── DynamicNeuralGNN (SIREN / MLP encoder) ──────────────────────────

class TestDynamicNeuralGNN:
    @pytest.fixture
    def tiny_mlp(self):
        """Graph spec for a 2 → 3 → 1 MLP."""
        act_num = torch.tensor([2, 3, 1])
        return get_graph_spec(act_num)

    def _make_input(self, graph_spec, batch=2):
        act_num = graph_spec[0]
        num_edges = sum(
            act_num[i] * act_num[i + 1] for i in range(len(act_num) - 1)
        )
        num_biases = sum(act_num[1:])
        return torch.randn(batch, num_edges), torch.randn(batch, num_biases)

    def test_forward_shape(self, tiny_mlp):
        emb_dim = 16
        model = DynamicNeuralGNN(
            graph_spec=tiny_mlp,
            fourier_dim=8, fourier_scale=10,
            rnn_mode='gru', rnn_layer=1, emb_dim=emb_dim,
        )
        edges, biases = self._make_input(tiny_mlp)
        out = model((edges, biases))
        # Last layer has 1 node → output is (B, 1 * emb_dim)
        assert out.shape == (2, 1 * emb_dim)

    def test_rnn_mode_rnn(self, tiny_mlp):
        model = DynamicNeuralGNN(
            graph_spec=tiny_mlp,
            fourier_dim=8, fourier_scale=10,
            rnn_mode='rnn', rnn_layer=1, emb_dim=16,
        )
        edges, biases = self._make_input(tiny_mlp)
        out = model((edges, biases))
        assert out.shape == (2, 16)

    def test_multi_rnn_layer(self, tiny_mlp):
        model = DynamicNeuralGNN(
            graph_spec=tiny_mlp,
            fourier_dim=8, fourier_scale=10,
            rnn_mode='gru', rnn_layer=3, emb_dim=16,
        )
        edges, biases = self._make_input(tiny_mlp)
        out = model((edges, biases))
        assert out.shape == (2, 16)

    def test_gradient_flow(self, tiny_mlp):
        model = DynamicNeuralGNN(
            graph_spec=tiny_mlp,
            fourier_dim=8, fourier_scale=10,
            rnn_mode='gru', rnn_layer=1, emb_dim=16,
        )
        edges, biases = self._make_input(tiny_mlp, batch=1)
        out = model((edges, biases))
        out.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


# ── DynamicNeuralGNN_Trans (ViT encoder) ────────────────────────────

class TestDynamicNeuralGNNTrans:
    @pytest.fixture
    def tiny_vit(self):
        return get_vit_graph_spec(
            num_block=1, dim=4, dim_head=2, head=2,
            mlp_dim=8, num_cls=3, patch_size=2, channels=1,
        )

    def test_forward_shape(self, tiny_vit):
        emb_dim = 8
        model = DynamicNeuralGNN_Trans(
            graph_spec=tiny_vit,
            fourier_dim=4, fourier_scale=10,
            rnn_mode='gru', emb_dim=emb_dim,
        )
        ne, nn_ = _count_edges_nodes_for_vit(tiny_vit)
        edges = torch.randn(2, ne)
        nodes = torch.randn(2, nn_)
        out = model((edges, nodes))
        assert out.shape == (2, tiny_vit['num_cls'] * emb_dim)

    def test_gradient_flow(self, tiny_vit):
        model = DynamicNeuralGNN_Trans(
            graph_spec=tiny_vit,
            fourier_dim=4, fourier_scale=10,
            rnn_mode='gru', emb_dim=8,
        )
        ne, nn_ = _count_edges_nodes_for_vit(tiny_vit)
        edges = torch.randn(1, ne)
        nodes = torch.randn(1, nn_)
        out = model((edges, nodes))
        out.sum().backward()
        with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total = sum(1 for _ in model.parameters())
        assert with_grad > total * 0.8, (
            f"Only {with_grad}/{total} params received gradients"
        )


# ── ViT_Gen_Predictor (full pipeline) ───────────────────────────────

class TestViTGenPredictor:
    # ViT_Gen_Predictor hardcodes head input = 10 * emb_dim, so num_cls must be 10.
    def test_forward_shape_and_range(self):
        spec = get_vit_graph_spec(
            num_block=1, dim=4, dim_head=2, head=2,
            mlp_dim=8, num_cls=10, patch_size=2, channels=1,
        )
        model = ViT_Gen_Predictor(
            graph_spec=spec,
            fourier_dim=4, fourier_scale=10,
            rnn_mode='gru', emb_dim=8,
            head_dim=16, sigmoid=True,
        )
        ne, nn_ = _count_edges_nodes_for_vit(spec)
        edges = torch.randn(2, ne)
        nodes = torch.randn(2, nn_)
        out = model((edges, nodes))
        assert out.shape == (2, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_no_sigmoid(self):
        spec = get_vit_graph_spec(
            num_block=1, dim=4, dim_head=2, head=2,
            mlp_dim=8, num_cls=10, patch_size=2, channels=1,
        )
        model = ViT_Gen_Predictor(
            graph_spec=spec,
            fourier_dim=4, fourier_scale=10,
            rnn_mode='gru', emb_dim=8,
            head_dim=16, sigmoid=False,
        )
        ne, nn_ = _count_edges_nodes_for_vit(spec)
        out = model((torch.randn(2, ne), torch.randn(2, nn_)))
        assert out.shape == (2, 1)
