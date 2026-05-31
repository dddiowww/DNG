import torch
import math
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange

class GaussianFourierFeatureTransform(nn.Module):
    def __init__(self, in_channels=1, mapping_size=128, scale=10):
        super().__init__()
        self.register_buffer("b", torch.randn((in_channels, mapping_size)) * scale)

    def forward(self, x):
        ch_dim = 1
        x = x.unsqueeze(ch_dim)
        x = (x.transpose(ch_dim, -1) @ self.b).transpose(ch_dim, -1)
        x = 2 * math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=ch_dim)

class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_map = nn.Linear(input_dim, 3 * hidden_dim)
        self.hidden_map = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden_state):
        gate_x = self.input_map(x)
        gate_h = self.hidden_map(hidden_state)
        
        i_r, i_z, i_n = gate_x.tensor_split(3, dim=-1)
        h_r, h_z, h_n = gate_h.tensor_split(3, dim=-1)
        
        r = F.sigmoid(i_r + h_r)
        z = F.sigmoid(i_z + h_z)
        n = F.tanh(i_n + (r * h_n))
        hy = (1 - z) * n + z * hidden_state
        return hy

class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.hidden_lin = nn.Linear(hidden_dim, hidden_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, h_tm1):
        node_value = F.relu(self.input_lin(x) + self.hidden_lin(h_tm1))
        return node_value

RNN = {'gru':GRUCell, 'rnn':RNNCell}
class DynamicNeuralGNN(nn.Module):
    def __init__(
        self, 
        graph_spec, 
        fourier_dim, 
        fourier_scale, 
        rnn_mode, 
        rnn_layer, 
        emb_dim, 
        bi_directed=False, 
        drop=0.0,
        # only for ablation study:
        positional_embedding=False,
        activation_embedding=False,
    ):
        super(DynamicNeuralGNN, self).__init__()
        self.node_num, self.node_id, self.n_e_connection = graph_spec
        self.rnn_layer = rnn_layer
        self.bi_directed = bi_directed
        self.drop = drop
        self.pe = positional_embedding
        self.ae = activation_embedding

        self.fourier_layer = GaussianFourierFeatureTransform(mapping_size=fourier_dim, scale=fourier_scale)
        if positional_embedding:
            self.pos_emb = nn.Parameter(torch.randn([len(self.node_id), fourier_dim*2]))
        if activation_embedding:
            self.act_emb = nn.Parameter(torch.randn([2, fourier_dim*2])) # only for sin activation function and no activation function

        for i in range(len(self.node_num)):
            self.add_module(f'split_edge_{i}', Rearrange('b (n e) c -> b n e c', n=self.node_num[i]))
        
        self.input_init_node = nn.Parameter(torch.randn([self.node_num[0], emb_dim]))
        for l in range(rnn_layer):
            self.add_module(f'rnn_{l}', RNN[rnn_mode](emb_dim, emb_dim))
            for i in range(len(self.node_id)):
                self.add_module(f'norm_{l}_{i}', nn.LayerNorm(emb_dim))
                self.add_module(f'node_lin_{l}_{i}', nn.Linear(emb_dim, emb_dim))
                self.add_module(f'edge_lin_{l}_{i}', nn.Linear(fourier_dim*2, emb_dim))
                self.add_module(f'input_lin_{l}_{i}', nn.Linear(fourier_dim*2, emb_dim))

    def forward(self, data):
        e = self.fourier_layer(data[0]).transpose(1,-1)
        b = self.fourier_layer(data[1]).transpose(1,-1)

        if self.pe:
            for layer in range(len(self.node_id)):
                b[:, self.node_id[layer]] = b[:, self.node_id[layer]] + self.pos_emb[layer].unsqueeze(0).unsqueeze(0)
        if self.ae:
            for layer in range(len(self.node_id)):
                b[:, self.node_id[layer]] = b[:, self.node_id[layer]] + self.act_emb[1 if layer==len(self.node_id)-1 else 0].unsqueeze(0).unsqueeze(0)
        
        for l in range(self.rnn_layer):
            hidden_node = self.input_init_node.unsqueeze(0).repeat([b.shape[0], 1, 1]) # (B, 2, emb_dim)
            next_iter_input_list = []
            for i in range(len(self.node_id)):
                edges = e[:, self.n_e_connection[i+1] if i==len(self.node_id)-1 else self.n_e_connection[i+1][0]] # (B, n_l*n_lm1, f_dim*2)
                hidden_edge = getattr(self, f'split_edge_{i+1}')(getattr(self, f'edge_lin_{l}_{i}')(edges)) # (B, n_l, n_lm1, emb_dim)
                hidden_state = torch.sum(hidden_edge * getattr(self, f'node_lin_{l}_{i}')(hidden_node).unsqueeze(1), dim=-2) # (B, n_l, emb_dim)
                node_self_value = getattr(self, f'input_lin_{l}_{i}')(b[:, self.node_id[i]] if l==0 else next_iter_input[:, self.node_id[i]]) # (B, n_l, emb_dim)
                hidden_node = getattr(self, f'rnn_{l}')(node_self_value, hidden_state) # (B, n_l, emb_dim)
                hidden_node = getattr(self, f'norm_{l}_{i}')(hidden_node)
                hidden_node = F.dropout(hidden_node, p=self.drop, training=self.training, inplace=False)
                next_iter_input_list.append(hidden_node)
            next_iter_input = torch.cat(next_iter_input_list, dim=1) # (B, n-2, emb_dim)
        
        x = hidden_node.flatten(start_dim=1) # (B, 3 or 1 * emb_dim)
        return x

class LatentGen(nn.Module):
    def __init__(self, latent_num, in_ch, out_ch, ds='cifar'):
        super(LatentGen, self).__init__()
        self.latent_num = latent_num
      
        self.latent_init = nn.Parameter(torch.randn([latent_num, in_ch]))
        self.src_fusion = nn.Sequential(nn.Linear(in_ch * (4 if 'cifar' in ds else 2), 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, out_ch))
            
    def get_rnn_ft(self, x):
        return x.flatten(start_dim=1) # (B, in_ch*3 or in_ch)
    
    def forward(self, x):
        latent_init = self.latent_init.unsqueeze(0).repeat([x.shape[0], 1, 1]) # (B, latent_num, in_ch/out_ch)
        rnn_src = self.get_rnn_ft(x)
        latent = self.src_fusion(torch.cat([latent_init, rnn_src.unsqueeze(1).repeat([1, self.latent_num, 1])], dim=-1)) # (B, latent_num, out_ch)
        return latent

class ConvDecoder64(nn.Module):
    def __init__(self, in_ch, ds='cifar', img_aug_size=6):
        super(ConvDecoder64, self).__init__()
        self.img_aug_size = img_aug_size
        self.conv = nn.Sequential(nn.ConvTranspose2d(in_ch, 256, 4, stride=2, padding=1),  # (16, 16)
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(256, 3 if 'cifar' in ds else 1, 4, stride=2, padding=1),  # (32, 32)
                                  )
    def forward(self, latent):
        latent = rearrange(latent, 'b (v h w) c -> (b v) c h w ', h=8, w=8)
        out = self.conv(latent) # (B*v, img_ch, h, w)
        out = rearrange(out, '(b v) c h w -> b (v c) h w', v=self.img_aug_size)
        return out

class ConvDecoder49(nn.Module):
    def __init__(self, in_ch, ds='mnist', img_aug_size=6):
        super(ConvDecoder49, self).__init__()
        self.img_aug_size = img_aug_size
        self.conv = nn.Sequential(nn.ConvTranspose2d(in_ch, 256, 4, stride=2, padding=1),  # (14, 14)
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(256, 3 if 'cifar' in ds else 1, 4, stride=2, padding=1),  # (28, 28)
                                  )
    def forward(self, latent):
        latent = rearrange(latent, 'b (v h w) c -> (b v) c h w ', h=7, w=7)
        out = self.conv(latent) # (B*v, img_ch, h, w)
        out = rearrange(out, '(b v) c h w -> b (v c) h w', v=self.img_aug_size)
        return out


class Autoencoder(nn.Module):
    def __init__(
        self, 
        graph_spec, 
        fourier_dim, 
        fourier_scale, 
        rnn_mode, 
        rnn_layer, 
        emb_dim, 
        latent_dim, 
        latent_size, 
        drop=0.2, 
        ds='cifar', 
        img_aug_size=6,
        # only for ablation study:
        pe=False,
        ae=False,
    ):
        super(Autoencoder, self).__init__()
        self.encoder = DynamicNeuralGNN(graph_spec, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim, drop=drop,
                                        positional_embedding=pe, activation_embedding=ae)
        self.latent_gen = LatentGen(latent_size*img_aug_size, emb_dim, latent_dim, ds)
        if 'cifar' in ds:
            self.decoder = ConvDecoder64(latent_dim, ds, img_aug_size)
        else:
            self.decoder = ConvDecoder49(latent_dim, ds, img_aug_size)
    
    def encode(self, data):
        x = self.encoder(data)
        x = self.latent_gen(x)
        return x
    
    def forward(self, data):
        x = self.encode(data)
        x = self.decoder(x)
        return x

class Autoencoder_NonSpatial(nn.Module):
    def __init__(
        self, 
        graph_spec, 
        fourier_dim, 
        fourier_scale, 
        rnn_mode, 
        rnn_layer, 
        emb_dim,
        dec_dim=512,
        drop=0.2, 
        ds='cifar'
    ):
        super(Autoencoder_NonSpatial, self).__init__()
        self.ds = ds
        self.encoder = DynamicNeuralGNN(graph_spec, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim, drop=drop)
        self.decoder = nn.Sequential(nn.Linear(emb_dim, dec_dim),
                                    #  nn.ReLU(),
                                    #  nn.Linear(dec_dim, dec_dim),
                                     nn.ReLU(),
                                     nn.Linear(dec_dim, 32*32 if 'cifar' in ds else 28*28)
                                    )
    
    def encode(self, data):
        x = self.encoder(data)
        return x
    
    def forward(self, data):
        x = self.encode(data)
        x = rearrange(x, 'b (c d)-> b c d', c = 3 if 'cifar' in self.ds else 1)
        x = self.decoder(x)
        x = rearrange(x, 'b c (h w)-> b c h w', h = 32 if 'cifar' in self.ds else 28)
        return x

class DNG_Classifier(nn.Module):
    def __init__(
        self,
        graph_spec, 
        fourier_dim, 
        fourier_scale, 
        rnn_mode, 
        rnn_layer, 
        emb_dim, 
        drop=0.2, 
        cls_dim=128, 
        cls_drop=0.5, 
        ds='cifar',
        # only for ablation study:
        pe=False,
        ae=False,
    ):
        super(DNG_Classifier, self).__init__()
        self.encoder = DynamicNeuralGNN(graph_spec, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim, drop=drop,
                                        positional_embedding=pe, activation_embedding=ae)
        self.classifier = nn.Sequential(nn.Linear((3 if 'cifar' in ds else 1) * emb_dim, cls_dim),
                                        nn.ReLU(),
                                        nn.Dropout(cls_drop),
                                        nn.Linear(cls_dim, cls_dim),
                                        nn.ReLU(),
                                        nn.Dropout(cls_drop),
                                        nn.Linear(cls_dim, (10 if '100' not in ds else 100))
                                        )
    
    def forward(self, data):
        x = self.encoder(data)
        x = self.classifier(x)
        return x

class Mapping(nn.Module):
    def __init__(self, ds, input_dim, hidden_dim, drop=0.0):
        super().__init__()
        self.ds = ds
        self.mapping = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, 32 * 32 if 'cifar' in ds else 28 * 28))
    def forward(self, x):
        x = rearrange(x, 'b (c d) -> b c d', c=3 if 'cifar' in self.ds else 1)
        x = self.mapping(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=32 if 'cifar' in self.ds else 28)
        return x

class DNGEncoderMapping(nn.Module):
    def __init__(
        self, 
        graph_spec, 
        fourier_dim, 
        fourier_scale, 
        rnn_mode, 
        rnn_layer, 
        emb_dim, 
        drop=0.2, 
        ds='cifar', 
    ):
        super(DNGEncoderMapping, self).__init__()
        self.encoder = DynamicNeuralGNN(graph_spec, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim, drop=drop)
        self.decoder = Mapping(ds, emb_dim, 512)
    
    def forward(self, data):
        x = self.encoder(data)
        x = self.decoder(x)
        return x

class HyperNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.mapping = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, out_dim))
    def forward(self, x):
        return self.mapping(x)

class Autoencoder_inr2inr(nn.Module):
    def __init__(
        self, 
        graph_spec, 
        fourier_dim, 
        fourier_scale, 
        rnn_mode, 
        rnn_layer, 
        emb_dim, 
        latent_dim, 
        latent_size,
        weight_num,
        bias_num,
        drop=0.2, 
        ds='cifar',
    ):
        super(Autoencoder_inr2inr, self).__init__()
        self.encoder = DynamicNeuralGNN(graph_spec, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim, drop=drop)
        self.latent_gen = LatentGen(latent_size, emb_dim, latent_dim, ds)
        self.weights_decoder = HyperNetwork(latent_dim, 512, weight_num)
        self.biases_decoder = HyperNetwork(latent_dim, 512, bias_num)
    
    def encode(self, data):
        x = self.encoder(data)
        x = self.latent_gen(x)
        return x
    
    def forward(self, data):
        x = self.encode(data)
        x = x.flatten(start_dim=1)
        weights = self.weights_decoder(x)
        biases = self.biases_decoder(x)
        return weights, biases

class DynamicNeuralGNN_MultiHead(nn.Module):
    def __init__(
        self, 
        graph_spec, 
        layer_type,
        edge_in, 
        fourier_dim, 
        fourier_scale, 
        rnn_mode, 
        rnn_layer, 
        emb_dim, 
        att_dim, 
        drop=0.0, 
        layer_emb=False
    ):
        super(DynamicNeuralGNN_MultiHead, self).__init__()
        self.act_num, self.node_id, self.n_e_connection = graph_spec
        self.layer_type = layer_type
        self.edge_in = edge_in
        self.rnn_layer = rnn_layer
        self.drop = drop
        self.layer_emb = layer_emb
        self.head = max(edge_in)

        self.fourier_layer = GaussianFourierFeatureTransform(in_channels=1, mapping_size=fourier_dim, scale=fourier_scale)
        self.register_buffer("paddings", torch.zeros([1, 1, 1, 1]))

        for i in range(len(self.act_num)):
            self.add_module(f'split_edge_{i}', Rearrange('b h (n e) c -> b h n e c', n=self.act_num[i]))
        
        self.input_init_node = nn.Parameter(torch.randn([self.act_num[0], emb_dim]))
        self.act_emb = nn.Parameter(torch.randn([2, fourier_dim*2]))
        if layer_emb:
            self.layer_embeddings = nn.Parameter(torch.randn([2, fourier_dim*2]))
        for l in range(rnn_layer):
            self.add_module(f'rnn_{l}', RNN[rnn_mode](emb_dim, emb_dim))
            for i in range(len(self.node_id)):
                self.add_module(f'norm_{l}_{i}', nn.LayerNorm(emb_dim))
                self.add_module(f'src_node_lin_{l}_{i}', nn.Linear(emb_dim, att_dim * self.head))
                self.add_module(f'edge_lin_{l}_{i}', nn.Linear(fourier_dim*2, att_dim))
                self.add_module(f'mapping_{l}_{i}', nn.Sequential(nn.Linear(att_dim * self.head, emb_dim),
                                                                  nn.ReLU(),
                                                                  nn.Linear(emb_dim, emb_dim)
                                                                  ))

                self.add_module(f'dst_node_lin_{l}_{i}', nn.Linear(fourier_dim*2, emb_dim))
        self.split_head = Rearrange('b n (h d) -> b h n d', h=self.head)
        self.combine_head = Rearrange('b h n d -> b n (h d)')

    def forward(self, data, activations):
        e, b = [], []
        for l in range(len(self.layer_type)):
            if 'conv' in self.layer_type[l]:
                f_e = rearrange(self.fourier_layer(data[0][l]), 'b d h n -> b h n d')
                if self.layer_emb:
                    b.append(self.fourier_layer(data[1][l]).transpose(1,-1) + self.layer_embeddings[0][None, None, ...])
                else:
                    b.append(self.fourier_layer(data[1][l]).transpose(1,-1))
            elif 'dense' in self.layer_type[l]:
                f_e = self.fourier_layer(data[0][l]).transpose(1,-1).unsqueeze(1)
                if self.layer_emb:
                    b.append(self.fourier_layer(data[1][l]).transpose(1,-1) + self.layer_embeddings[1][None, None, ...])
                else:
                    b.append(self.fourier_layer(data[1][l]).transpose(1,-1))
            
            if self.head > f_e.shape[1]:
                paddings = self.paddings.repeat([f_e.shape[0], self.head-f_e.shape[1], f_e.shape[2], f_e.shape[3]])
                e.append(torch.cat([f_e, paddings], dim=1))
            else:
                e.append(f_e)

        e = torch.cat(e, dim=2) # (B, max_edge_in, e_num, f_dim*2)
        b = torch.cat(b, dim=1) # (B, b_num, f_dim*2)
        b = b + self.act_emb[activations].unsqueeze(1)

        for l in range(self.rnn_layer):
            hidden_node = self.input_init_node.unsqueeze(0).repeat([b.shape[0], 1, 1]) # (B, n_0, emb_dim)
            next_layer_input_list = []
            for i in range(len(self.node_id)):
                edges = e[:, :, self.n_e_connection[i+1] if i==len(self.node_id)-1 else self.n_e_connection[i+1][0]] # (B, h, n_l*n_lm1, f_dim*2)
                hidden_edge = getattr(self, f'split_edge_{i+1}')(getattr(self, f'edge_lin_{l}_{i}')(edges)) # (B, h, n_l, n_lm1, att_dim)
                source_node = self.split_head(getattr(self, f'src_node_lin_{l}_{i}')(hidden_node)).unsqueeze(2) # (B, h, 1, n_lm1, att_dim)

                hidden_state = rearrange(hidden_edge * source_node, 'b h n m d -> b n m (h d)') # (B, n_l, n_lm1, h * att_dim)
                hidden_state = getattr(self, f'mapping_{l}_{i}')(hidden_state) # (B, n_l, n_lm1, emb_dim)
                hidden_state = torch.sum(hidden_state, dim=-2) # (B, n_l, emb_dim)

                dst_node_value = getattr(self, f'dst_node_lin_{l}_{i}')(b[:, self.node_id[i]] if l==0 else next_layer_input[:, self.bias_id[i]]) # (B, n_l, emb_dim)
                hidden_node = getattr(self, f'rnn_{l}')(dst_node_value, hidden_state) # (B, n_l, emb_dim)
        
                hidden_node = F.dropout(getattr(self, f'norm_{l}_{i}')(hidden_node), p=self.drop, training=self.training, inplace=False)
                next_layer_input_list.append(hidden_node)
            next_layer_input = torch.cat(next_layer_input_list, dim=1) # (B, n-2, emb_dim)

        x = hidden_node.flatten(start_dim=1)
        return x

class Gen_Predictor(nn.Module):
    def __init__(
        self, 
        graph_spec, 
        layer_type,
        edge_in, 
        fourier_dim, 
        fourier_scale, 
        rnn_mode, 
        rnn_layer, 
        emb_dim, 
        att_dim, 
        head_dim, 
        head_drop=0.0, 
        sigmoid=True
    ):
        super(Gen_Predictor, self).__init__()
        self.encoder = DynamicNeuralGNN_MultiHead(graph_spec, layer_type, edge_in, fourier_dim, fourier_scale, rnn_mode, rnn_layer, emb_dim, att_dim)
        self.head = nn.Sequential(nn.Linear(10 * emb_dim, head_dim),
                                  nn.ReLU(),
                                  nn.Dropout(head_drop),
                                  nn.Linear(head_dim, head_dim),
                                  nn.ReLU(),
                                  nn.Dropout(head_drop),
                                  nn.Linear(head_dim, 1),
                                  nn.Sigmoid() if sigmoid else nn.Identity()
                                  )
    
    def forward(self, data, act):
        x = self.encoder(data, act)
        x = self.head(x)
        return x

class DynamicNeuralGNN_Trans(nn.Module):
    def __init__(
        self, 
        graph_spec, 
        fourier_dim, 
        fourier_scale, 
        rnn_mode, 
        emb_dim, 
        drop=0.0
    ):
        super(DynamicNeuralGNN_Trans, self).__init__()
        self.graph_spec = graph_spec
        self.drop = drop

        self.fourier_layer = GaussianFourierFeatureTransform(mapping_size=fourier_dim, scale=fourier_scale)

        self.add_module(f'to_emb_split_edge', Rearrange('b (n e) c -> b n e c', n = graph_spec['dim'])) # e = # of previous layer nodes
        self.add_module(f'to_qkv_split_edge', Rearrange('b (n e) c -> b n e c', n = graph_spec['block_layer_node_num'][1]))
        self.add_module(f'to_out_split_edge', Rearrange('b (n e) c -> b n e c', n = graph_spec['dim']))
        self.add_module(f'fc1_split_edge', Rearrange('b (n e) c -> b n e c', n = graph_spec['mlp_dim']))
        self.add_module(f'fc2_split_edge', Rearrange('b (n e) c -> b n e c', n = graph_spec['dim']))
        self.add_module(f'head_split_edge', Rearrange('b (n e) c -> b n e c', n = graph_spec['num_cls']))

        self.input_init_node = nn.Parameter(torch.randn([graph_spec['num_init_node'], emb_dim]))
        self.qkv_comp_dict = torch.nn.ParameterDict()
        self.norm_dict = torch.nn.ParameterDict()

        self.norm_dict[f'init_norm_std'] = nn.Parameter(torch.randn([1, 1, emb_dim]))
        self.norm_dict[f'init_norm_mean'] = nn.Parameter(torch.randn([1, 1, emb_dim]))
        self.add_module(f'to_emb_node_lin', nn.Linear(emb_dim, emb_dim))
        self.add_module(f'to_emb_edge_lin', nn.Linear(fourier_dim*2, emb_dim)) 
        self.add_module(f'to_emb_input_lin', nn.Linear(fourier_dim*2, emb_dim))
        self.add_module(f'to_emb_emb_norm', nn.LayerNorm(emb_dim))

        self.add_module(f'rnn', RNN[rnn_mode](emb_dim, emb_dim))
        for i in range(graph_spec['num_block']):
            for j in range(len(graph_spec['block_layer_node_num'])):
                self.add_module(f'node_lin_{i}_{j}', nn.Linear(emb_dim, emb_dim))
                self.add_module(f'emb_norm_{i}_{j}', nn.LayerNorm(emb_dim))

            self.norm_dict[f'att_norm_std_{i}'] = nn.Parameter(torch.randn([1, 1, emb_dim]))
            self.norm_dict[f'att_norm_mean_{i}'] = nn.Parameter(torch.randn([1, 1, emb_dim]))
            self.add_module(f'qkv_edge_lin_{i}', nn.Linear(fourier_dim*2, emb_dim))
            self.add_module(f'out_edge_lin_{i}', nn.Linear(fourier_dim*2, emb_dim))
            self.qkv_comp_dict[f'qkv_comp_edges_{i}'] = nn.Parameter(torch.randn([graph_spec['dim_head'], graph_spec['dim_head'] * 3, emb_dim]))

            self.norm_dict[f'fc_norm_std_{i}'] = nn.Parameter(torch.randn([1, 1, emb_dim]))
            self.norm_dict[f'fc_norm_mean_{i}'] = nn.Parameter(torch.randn([1, 1, emb_dim]))
            self.add_module(f'fc1_edge_lin_{i}', nn.Linear(fourier_dim*2, emb_dim))
            self.add_module(f'fc2_edge_lin_{i}', nn.Linear(fourier_dim*2, emb_dim))
            self.add_module(f'fc1_input_lin_{i}', nn.Linear(fourier_dim*2, emb_dim))
            self.add_module(f'fc2_input_lin_{i}', nn.Linear(fourier_dim*2, emb_dim))
            
        self.norm_dict[f'last_norm_std'] = nn.Parameter(torch.randn([1, 1, emb_dim]))
        self.norm_dict[f'last_norm_mean'] = nn.Parameter(torch.randn([1, 1, emb_dim]))
        self.add_module(f'last_emb_norm', nn.LayerNorm(emb_dim))

        self.add_module(f'head_node_lin', nn.Linear(emb_dim, emb_dim))
        self.add_module(f'head_edge_lin', nn.Linear(fourier_dim*2, emb_dim))
        self.add_module(f'head_input_lin', nn.Linear(fourier_dim*2, emb_dim))
        self.add_module(f'head_emb_norm', nn.LayerNorm(emb_dim))

    def forward(self, data):
        e = self.fourier_layer(data[0]).transpose(1,-1)
        b = self.fourier_layer(data[1]).transpose(1,-1)

        init_nodes = self.input_init_node.unsqueeze(0).repeat([b.shape[0], 1, 1]) # (B, d_init, emb_dim)
        init_nodes = init_nodes * self.norm_dict['init_norm_std'] + self.norm_dict['init_norm_mean']
        to_emb_edges = e[:, self.graph_spec['to_emb_edge_id']] # (B, d_init * d_emb, f_dim * 2)
        to_emb_edges = getattr(self, 'to_emb_split_edge')(getattr(self, 'to_emb_edge_lin')(to_emb_edges)) # (B, d_emb, d_init, emb_dim)
        to_emb_hidden_state = torch.sum(to_emb_edges * getattr(self, 'to_emb_node_lin')(init_nodes).unsqueeze(1), dim=-2) # (B, d_emb, emb_dim)
        to_emb_input_nodes = getattr(self, 'to_emb_input_lin')(b[:, self.graph_spec['to_emb_node_id']]) # (B, d_emb, emb_dim)
        to_emb_nodes = getattr(self, 'rnn')(to_emb_input_nodes, to_emb_hidden_state) # (B, d_emb, emb_dim)
        to_emb_nodes = getattr(self, 'to_emb_emb_norm')(to_emb_nodes)

        for i in range(self.graph_spec['num_block']):
            start_nodes = to_emb_nodes if i == 0 else block_last_nodes
            start_nodes_norm = start_nodes * self.norm_dict[f'att_norm_std_{i}'] + self.norm_dict[f'att_norm_mean_{i}'] # layernorm -> add d_emb nodes and edges
            qkv_edges = e[:, self.graph_spec[f'qkv_edge_id_{i}']] # (B, d_emb * d_qkv, f_dim * 2)
            qkv_edges = getattr(self, 'to_qkv_split_edge')(getattr(self, f'qkv_edge_lin_{i}')(qkv_edges)) # (B, d_qkv, d_emb, emb_dim)
            qkv_nodes = torch.sum(qkv_edges * getattr(self, f'node_lin_{i}_0')(start_nodes_norm).unsqueeze(1), dim=-2) # (B, d_qkv, emb_dim)
            qkv_nodes = getattr(self, f'emb_norm_{i}_1')(qkv_nodes)
            qkv_comp_edges = self.qkv_comp_dict[f'qkv_comp_edges_{i}'].unsqueeze(0).repeat([qkv_nodes.shape[0], 1, 1, 1]) # (B, d_head, d_head * 3, emb_dim)
            z_nodes = []
            for ids in self.graph_spec['head_ids']:
                head_qkv_nodes = qkv_nodes[:, ids] # (B, d_head * 3 = d_qkv / h, emb_dim)
                head_z_nodes = torch.sum(qkv_comp_edges * getattr(self, f'node_lin_{i}_1')(head_qkv_nodes).unsqueeze(1), dim=-2) # (B, d_head, emb_dim)
                z_nodes.append(head_z_nodes)
            z_nodes = torch.cat(z_nodes, dim=1) # (B, d_head * h, emb_dim)
            z_nodes = getattr(self, f'emb_norm_{i}_2')(z_nodes)
            out_edges = e[:, self.graph_spec[f'out_edge_id_{i}']] # (B, d_head * h * d_emb, f_dim * 2)
            out_edges = getattr(self, 'to_out_split_edge')(getattr(self, f'out_edge_lin_{i}')(out_edges)) # (B, d_emb, d_head * h, emb_dim)
            out_nodes = torch.sum(out_edges * getattr(self, f'node_lin_{i}_2')(z_nodes).unsqueeze(1), dim=-2) # (B, d_emb, emb_dim)
            out_nodes = getattr(self, f'emb_norm_{i}_3')(out_nodes)
            out_nodes = out_nodes + start_nodes # residual connection

            fc_start_nodes = out_nodes * self.norm_dict[f'fc_norm_std_{i}'] + self.norm_dict[f'fc_norm_mean_{i}'] # (B, d_emb, emb_dim)
            fc1_edges = e[:, self.graph_spec[f'fc1_edge_id_{i}']] # (B, d_emb * d_ff, f_dim * 2)
            fc1_edges = getattr(self, 'fc1_split_edge')(getattr(self, f'fc1_edge_lin_{i}')(fc1_edges)) # (B, d_ff, d_emb, emb_dim)
            fc1_hidden_state = torch.sum(fc1_edges * getattr(self, f'node_lin_{i}_3')(fc_start_nodes).unsqueeze(1), dim=-2) # (B, d_ff, emb_dim)
            fc1_input_nodes = getattr(self, f'fc1_input_lin_{i}')(b[:, self.graph_spec[f'fc1_node_id_{i}']]) # (B, d_ff, emb_dim)
            fc1_nodes = getattr(self, 'rnn')(fc1_input_nodes, fc1_hidden_state) # (B, d_ff, emb_dim)
            fc1_nodes = getattr(self, f'emb_norm_{i}_4')(fc1_nodes)
            # fc1_nodes = F.dropout(fc1_nodes, p=self.drop, training=self.training, inplace=False)

            fc2_edges = e[:, self.graph_spec[f'fc2_edge_id_{i}']] # (B, d_ff * d_emb, f_dim * 2)
            fc2_edges = getattr(self, 'fc2_split_edge')(getattr(self, f'fc2_edge_lin_{i}')(fc2_edges)) # (B, d_emb, d_ff, emb_dim)
            fc2_hidden_state = torch.sum(fc2_edges * getattr(self, f'node_lin_{i}_4')(fc1_nodes).unsqueeze(1), dim=-2) # (B, d_emb, emb_dim)
            fc2_input_nodes = getattr(self, f'fc2_input_lin_{i}')(b[:, self.graph_spec[f'fc2_node_id_{i}']]) # (B, d_emb, emb_dim)
            block_last_nodes = getattr(self, 'rnn')(fc2_input_nodes, fc2_hidden_state) # (B, d_emb, emb_dim)
            fc1_nodes = getattr(self, f'emb_norm_{i}_5')(fc1_nodes)
            # fc1_nodes = F.dropout(fc1_nodes, p=self.drop, training=self.training, inplace=False)
            block_last_nodes = block_last_nodes + out_nodes
        
        block_last_nodes = block_last_nodes * self.norm_dict['last_norm_std'] + self.norm_dict['last_norm_mean']
        block_last_nodes = getattr(self, 'last_emb_norm')(block_last_nodes)

        head_edges = e[:, self.graph_spec['head_edge_id']] # (B, d_emb * d_cls, f_dim * 2)
        head_edges = getattr(self, 'head_split_edge')(getattr(self, 'head_edge_lin')(head_edges)) # (B, d_cls, d_emb, emb_dim)
        head_hidden_state = torch.sum(head_edges * getattr(self, 'head_node_lin')(block_last_nodes).unsqueeze(1), dim=-2) # (B, d_cls, emb_dim)
        head_input_nodes = getattr(self, 'head_input_lin')(b[:, self.graph_spec['head_node_id']]) # (B, d_cls, emb_dim)
        head_nodes = getattr(self, 'rnn')(head_input_nodes, head_hidden_state) # (B, d_cls, emb_dim)
        head_nodes = getattr(self, 'head_emb_norm')(head_nodes)
        
        x = head_nodes.flatten(start_dim=1) # (B, 10 * emb_dim)
        return x

class ViT_Gen_Predictor(nn.Module):
    def __init__(
        self, 
        graph_spec, 
        fourier_dim, 
        fourier_scale, 
        rnn_mode, 
        emb_dim, 
        head_dim, 
        head_drop=0.0, 
        sigmoid=True
    ):
        super(ViT_Gen_Predictor, self).__init__()
        self.encoder = DynamicNeuralGNN_Trans(graph_spec, fourier_dim, fourier_scale, rnn_mode, emb_dim)
        self.head = nn.Sequential(nn.Linear(10 * emb_dim, head_dim),
                                  nn.ReLU(),
                                  nn.Dropout(head_drop),
                                  nn.Linear(head_dim, head_dim),
                                  nn.ReLU(),
                                  nn.Dropout(head_drop),
                                  nn.Linear(head_dim, 1),
                                  nn.Sigmoid() if sigmoid else nn.Identity()
                                  )
    
    def forward(self, data):
        x = self.encoder(data)
        x = self.head(x)
        return x


# =====================================================================
# Models for heterogeneous CNN Wild Park dataset
# =====================================================================

class DNG_Park_Encoder(nn.Module):
    """
    DNG encoder for heterogeneous CNN architectures (CNN Wild Park).
    Uses shared weights across all NN layers so that models with different
    numbers of layers / channels / kernel sizes can be processed uniformly.
    """
    def __init__(
        self,
        fourier_dim,
        fourier_scale,
        rnn_mode,
        rnn_layer,
        emb_dim,
        att_dim,
        max_h,
        n_input_nodes=3,
        n_act_types=6,
        drop=0.0,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.att_dim = att_dim
        self.max_h = max_h
        self.rnn_layer = rnn_layer
        self.drop = drop
        self.f_dim = fourier_dim * 2

        self.fourier = GaussianFourierFeatureTransform(
            in_channels=1, mapping_size=fourier_dim, scale=fourier_scale
        )
        self.act_emb = nn.Embedding(n_act_types, self.f_dim)
        self.input_init_node = nn.Parameter(torch.randn(n_input_nodes, emb_dim))

        # Shared weights — reused at every NN layer within each RNN pass
        for rl in range(rnn_layer):
            self.add_module(f'rnn_{rl}', RNN[rnn_mode](emb_dim, emb_dim))
            self.add_module(f'edge_lin_{rl}', nn.Linear(self.f_dim, att_dim))
            self.add_module(f'src_node_lin_{rl}', nn.Linear(emb_dim, att_dim * max_h))
            self.add_module(f'mapping_{rl}', nn.Sequential(
                nn.Linear(att_dim * max_h, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
            ))
            self.add_module(f'norm_{rl}', nn.LayerNorm(emb_dim))
            # First RNN pass reads Fourier features (f_dim); later passes read embeddings (emb_dim)
            dst_in = self.f_dim if rl == 0 else emb_dim
            self.add_module(f'dst_node_lin_{rl}', nn.Linear(dst_in, emb_dim))

        # Residual message projection
        self.res_lin = nn.Linear(emb_dim, emb_dim)

    # ------------------------------------------------------------------
    def _forward_one_pass(
        self, rl, f_edges, f_nodes, edge_list, mask_list, lnn, offsets,
        total_nodes, res_index, res_mask, prev_all_emb, B, device,
    ):
        """One RNN pass through all NN layers."""
        all_emb = torch.zeros(B, total_nodes, self.emb_dim, device=device)
        hidden_node = self.input_init_node.unsqueeze(0).expand(B, -1, -1)
        all_emb[:, :offsets[1]] = hidden_node

        L = len(edge_list)
        for l in range(L):
            in_l = lnn[l]
            out_l = lnn[l + 1]
            H = edge_list[l].shape[1]  # actual h for this bucket

            # --- Edge features ---
            fe = f_edges[l]                                         # (B, H, pairs, f_dim)
            fe = getattr(self, f'edge_lin_{rl}')(fe)                # (B, H, pairs, att_dim)
            fe = fe.view(B, H, in_l, out_l, self.att_dim)
            fe = fe.permute(0, 1, 3, 2, 4)                         # (B, H, out, in, att_dim)

            # Pad H → max_h
            if H < self.max_h:
                pad_h = self.max_h - H
                fe = F.pad(fe, (0, 0, 0, 0, 0, 0, 0, pad_h))      # (B, max_h, out, in, att)
                mask_padded = F.pad(mask_list[l], (0, pad_h), value=False)
            else:
                mask_padded = mask_list[l]                          # (B, max_h)

            # --- Source node multi-head projection ---
            src = getattr(self, f'src_node_lin_{rl}')(hidden_node)  # (B, in, att*max_h)
            src = src.view(B, in_l, self.max_h, self.att_dim)
            src = src.permute(0, 2, 1, 3).unsqueeze(2)             # (B, max_h, 1, in, att)

            # --- Message ---
            msg = fe * src                                          # (B, max_h, out, in, att)
            emask = mask_padded.view(B, self.max_h, 1, 1, 1).float()
            msg = msg * emask
            msg = msg.permute(0, 2, 3, 1, 4).reshape(
                B, out_l, in_l, self.max_h * self.att_dim
            )                                                       # (B, out, in, max_h*att)

            msg = getattr(self, f'mapping_{rl}')(msg)               # (B, out, in, emb)
            msg = msg.sum(dim=2)                                    # (B, out, emb)

            # --- Destination (bias) features ---
            if rl == 0:
                dst = getattr(self, f'dst_node_lin_{rl}')(f_nodes[l])
            else:
                dst = getattr(self, f'dst_node_lin_{rl}')(
                    prev_all_emb[:, offsets[l + 1]:offsets[l + 2]]
                )

            # --- RNN update ---
            hidden_node = getattr(self, f'rnn_{rl}')(dst, msg)
            hidden_node = F.dropout(
                getattr(self, f'norm_{rl}')(hidden_node),
                p=self.drop, training=self.training,
            )
            all_emb[:, offsets[l + 1]:offsets[l + 2]] = hidden_node

            # --- Residual connections targeting this layer ---
            if res_index.shape[1] > 0:
                dst_start = offsets[l + 1]
                dst_end = offsets[l + 2]
                in_layer = (res_index[1] >= dst_start) & (res_index[1] < dst_end)
                if in_layer.any():
                    sel = in_layer.nonzero(as_tuple=True)[0]
                    src_emb = all_emb[:, res_index[0, sel]]         # (B, |sel|, emb)
                    r_msg = self.res_lin(src_emb)
                    r_msg = r_msg * res_mask[:, sel].unsqueeze(-1).float()
                    dst_local = (res_index[1, sel] - dst_start)
                    idx = dst_local.unsqueeze(0).unsqueeze(-1).expand(B, -1, self.emb_dim)
                    hidden_node = hidden_node.scatter_add(1, idx, r_msg)
                    all_emb[:, offsets[l + 1]:offsets[l + 2]] = hidden_node

        return hidden_node, all_emb

    # ------------------------------------------------------------------
    def forward(self, batch):
        edge_list = batch["edge"]              # List[L], (B, H, pairs_l)
        node_list = batch["node"]              # List[L], (B, out_l)
        mask_list = batch["mask"]              # List[L], (B, H)
        lnn       = batch["layer_node_num"]    # tuple (n0, n1, ..., nL)
        res_index = batch["residual_index"]    # (2, R)
        res_mask  = batch["residual_mask"]     # (B, R)
        act_ids   = batch["act_ids"]           # (B, L)

        B = edge_list[0].shape[0]
        L = len(edge_list)
        device = edge_list[0].device

        # Layer offsets for global node indexing
        offsets = [0]
        for n in lnn:
            offsets.append(offsets[-1] + n)
        total_nodes = offsets[-1]

        # Fourier transform edges: (B, H, pairs) → (B, H, pairs, f_dim)
        f_edges = []
        for l in range(L):
            fe = self.fourier(edge_list[l])     # (B, f_dim, H, pairs)
            f_edges.append(fe.permute(0, 2, 3, 1))

        # Fourier transform biases + per-layer activation embedding
        f_nodes = []
        for l in range(L):
            fn = self.fourier(node_list[l])     # (B, f_dim, out_l)
            fn = fn.transpose(1, 2)             # (B, out_l, f_dim)
            fn = fn + self.act_emb(act_ids[:, l]).unsqueeze(1)
            f_nodes.append(fn)

        # RNN passes
        prev_all_emb = None
        for rl in range(self.rnn_layer):
            hidden_node, all_emb = self._forward_one_pass(
                rl, f_edges, f_nodes, edge_list, mask_list, lnn, offsets,
                total_nodes, res_index, res_mask, prev_all_emb, B, device,
            )
            prev_all_emb = all_emb

        return hidden_node.flatten(start_dim=1)  # (B, n_last * emb_dim)


class Gen_Predictor_Park(nn.Module):
    """Generalization predictor for CNN Wild Park (heterogeneous CNNs)."""
    def __init__(
        self,
        fourier_dim,
        fourier_scale,
        rnn_mode,
        rnn_layer,
        emb_dim,
        att_dim,
        max_h,
        head_dim,
        n_input_nodes=3,
        n_act_types=6,
        n_classes=10,
        head_drop=0.0,
        sigmoid=True,
        drop=0.0,
    ):
        super().__init__()
        self.encoder = DNG_Park_Encoder(
            fourier_dim, fourier_scale, rnn_mode, rnn_layer,
            emb_dim, att_dim, max_h, n_input_nodes, n_act_types, drop,
        )
        self.head = nn.Sequential(
            nn.Linear(n_classes * emb_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(head_drop),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(head_drop),
            nn.Linear(head_dim, 1),
            nn.Sigmoid() if sigmoid else nn.Identity(),
        )

    def forward(self, batch):
        x = self.encoder(batch)
        return self.head(x)