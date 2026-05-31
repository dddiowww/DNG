import torch

def get_siren_shape(model_dict_path): 
    '''
    Get siren weight space shape ([W1.shape, W2.shape,...], [B1.shape, B2.shape,...])
    '''
    weight_spec = []
    bias_spec = []
    model_dict = torch.load(model_dict_path, map_location='cpu')
    for name, param in model_dict.items():
        if 'weight' in name:
            weight_spec.append(param.shape)
        if 'bias' in name:
            bias_spec.append(param.shape)
    return (weight_spec, bias_spec)

def get_act_num(siren_shape, input_layer=True, output_layer=True): 
    '''
    Get # of activations for each layer (may include input and output)
    '''
    act_num = []
    weight_shape, bias_shape = siren_shape
    for i in range(len(weight_shape)+1):
        if i != len(weight_shape):
            act_num.append(weight_shape[i][1])
        else:
            act_num.append(weight_shape[i-1][0])
    act_num = torch.tensor(act_num)
    if not input_layer:
        act_num = act_num[1:]
    if not output_layer:
        act_num = act_num[:-1]
    return act_num

def get_node_split_idx(act_num):
    all_idx = list(range(torch.sum(act_num)))
    node_idx = []
    split_idx = []
    for i in act_num:
        layer_node_idx = all_idx[:i]
        split = layer_node_idx[0]
        all_idx = all_idx[i:]
        node_idx.append(layer_node_idx)
        split_idx.append(split)
    split_idx.append(int(torch.sum(act_num)))
    return node_idx, split_idx

def get_layer_bias_id(act_num):
    act_num = act_num[1:]
    all_idx = list(range(torch.sum(act_num)))
    bias_idx = []
    for i in act_num:
        layer_bias_idx = all_idx[:i]
        all_idx = all_idx[i:]
        bias_idx.append(layer_bias_idx)
    return bias_idx

def get_edge_id(node_idx):
    '''
    Edge id and node connections in fully connected network
    Aggregated by each node and each layer
    '''
    edge_id = []
    side_nodes = []
    i = 0
    for l in range(len(node_idx)-1):
        layer_edge_id = []
        for node_id in node_idx[l]:
            # get edge id based on one node
            node_edge_id = [i+x for x in range(len(node_idx[l+1]))]
            layer_edge_id.append(node_edge_id)
            i += len(node_edge_id)
            # get nodes beside the edge
            connect_node = [[node_id, n] for n in node_idx[l+1]]
            side_nodes += connect_node
        edge_id.append(layer_edge_id)
    return edge_id, side_nodes

def get_layer_edge_id(edge_id, side_nodes):
    layer_edge_ids = []
    for l in range(len(edge_id)):
        layer_edge_id = torch.tensor(edge_id[l]).flatten()
        layer_edge_ids.append(layer_edge_id.tolist())
    
    layer_side_nodes = []
    for l in range(len(layer_edge_ids)):
        l_side_nodes = []
        for i in layer_edge_ids[l]:
            l_side_nodes += side_nodes[i]
        layer_side_nodes.append(l_side_nodes)
    return layer_edge_ids, layer_side_nodes

def get_layer_connections(act_num):
    node_num = []
    for num in act_num:
        node_num.append(list(range(num)))
    
    side_nodes = []
    for l in range(len(node_num)-1):
        layer_side_nodes = []
        for node_id in node_num[l]:
            connect_node = [[node_id, n] for n in node_num[l+1]]
            layer_side_nodes += connect_node
        layer_side_nodes = torch.tensor(layer_side_nodes).flatten()
        side_nodes.append(layer_side_nodes.tolist())
    return side_nodes

def get_node_edges(edge_id, node_idx, layer=False, separate=False):
    '''
    Find out which edges each node is connected to
    List([Node_0 connections], [Node_1 connections], ...)
    '''
    node_edge_connections = []
    for l in range(len(node_idx)):
        layer_n_e_connect = []
        for i, node in enumerate(node_idx[l]):
            if l==0:
                n_e_connect = edge_id[l][i]
            elif l==len(node_idx)-1:
                n_e_connect = [edge_id[l-1][e][i] for e in range(len(edge_id[l-1]))]
            else:
                if separate:
                    n_e_connect = [[edge_id[l-1][e][i] for e in range(len(edge_id[l-1]))], edge_id[l][i]]
                else:
                    n_e_connect = [edge_id[l-1][e][i] for e in range(len(edge_id[l-1]))] + edge_id[l][i]
            layer_n_e_connect.append(n_e_connect)
            if not layer:
                node_edge_connections.append(n_e_connect)
        if layer:
            node_edge_connections.append(layer_n_e_connect)
    return node_edge_connections

def get_node_edges_l_r(n_e_connections_l_s):
    out = []
    for l in range(len(n_e_connections_l_s)):
        l_out, left, right = [], [], []
        if l==0 or l==len(n_e_connections_l_s)-1:
            for node in n_e_connections_l_s[l]:
                l_out += node
            out.append(l_out)
        else:
            for node in n_e_connections_l_s[l]:
                left += node[0]
                right += node[1]
            out.append([left, right])
    return out

def get_edge_ft_bias_label(raw_data, act_num, label=True):
    '''
    edge feature is weight, bias feature is bias
    '''
    if label:
        (weight, bias), y = raw_data
    else:
        (weight, bias) = raw_data
    
    node_num = []
    for num in act_num:
        node_num.append(list(range(num)))

    edge_features = []
    bias_features = []
    for l in range(len(node_num)):
        if l != len(node_num)-1:
            bias_features += [bias[l].squeeze(0)[x] for x in node_num[l+1]]
        for n_i in node_num[l]:
            if l != len(node_num)-1:
                w_feature = [weight[l].squeeze(0)[x, n_i] for x in node_num[l+1]]
                edge_features += w_feature
    if label:
        return torch.tensor(edge_features), torch.tensor(bias_features), torch.tensor([y])
    else:
        return torch.tensor(edge_features), torch.tensor(bias_features)

def get_graph_spec(act_num):
    '''
    Get graph info for fully connected neural networks, with input layer and output layer
    '''
    node_id, _ = get_node_split_idx(act_num)
    edge_id, _ = get_edge_id(node_id)
    layer_bias_id = get_layer_bias_id(act_num)
    n_e_connection_l_s = get_node_edges(edge_id, node_id, layer=True, separate=True)
    n_e_connection = get_node_edges_l_r(n_e_connection_l_s)

    return act_num, layer_bias_id, n_e_connection

# Methods for Transformer Architecture
def split_head_indices(lst, h):
    n = len(lst)
    split_3 = [lst[i * (n // 3):(i + 1) * (n // 3)] for i in range(3)]

    split_h = []
    for part in split_3:
        m = len(part)
        split_h.append([part[i * (m // h):(i + 1) * (m // h)] for i in range(h)])

    result = []
    for i in range(h):
        merged = []
        for part in split_h:
            merged.extend(part[i])
        result.append(merged)

    return result

def get_vit_graph_spec(num_block, dim, dim_head, head, mlp_dim, num_cls, patch_size, channels=3):
    '''
    graph_spec stores basic ViT information &
    indices for the concrete edges, i.e. weights &
    indices for the concrete nodes, i.e. biases
    '''
    graph_spec = {}
    graph_spec['num_block'] = num_block
    graph_spec['dim'] = dim
    graph_spec['dim_head'] = dim_head
    graph_spec['head'] = head
    graph_spec['mlp_dim'] = mlp_dim
    graph_spec['num_cls'] = num_cls

    graph_spec['num_init_node'] = patch_size * patch_size * channels
    block_layer_node_num = [dim, dim_head * head * 3, dim_head * head, dim, mlp_dim, dim]
    graph_spec['block_layer_node_num'] = block_layer_node_num

    num_to_emb_edges = patch_size * patch_size * channels * dim
    graph_spec['to_emb_edge_id'] = list(range(num_to_emb_edges))
    edge_start = num_to_emb_edges

    num_to_emb_nodes = dim
    graph_spec['to_emb_node_id'] = list(range(num_to_emb_nodes))
    node_start = num_to_emb_nodes

    num_qkv_edges = block_layer_node_num[0] * block_layer_node_num[1]
    num_out_edges = block_layer_node_num[2] * block_layer_node_num[3]
    num_fc1_edges = block_layer_node_num[3] * block_layer_node_num[4]
    num_fc2_edges = block_layer_node_num[4] * block_layer_node_num[5]
    num_fc1_nodes = block_layer_node_num[4]
    num_fc2_nodes = block_layer_node_num[5]
    for i in range(num_block):
        graph_spec[f'qkv_edge_id_{i}'] = list(range(edge_start, edge_start + num_qkv_edges))
        edge_start = edge_start + num_qkv_edges

        graph_spec[f'out_edge_id_{i}'] = list(range(edge_start, edge_start + num_out_edges))
        edge_start = edge_start + num_out_edges

        graph_spec[f'fc1_edge_id_{i}'] = list(range(edge_start, edge_start + num_fc1_edges))
        edge_start = edge_start + num_fc1_edges

        graph_spec[f'fc2_edge_id_{i}'] = list(range(edge_start, edge_start + num_fc2_edges))
        edge_start = edge_start + num_fc2_edges

        graph_spec[f'fc1_node_id_{i}'] = list(range(node_start, node_start + num_fc1_nodes))
        node_start = node_start + num_fc1_nodes

        graph_spec[f'fc2_node_id_{i}'] = list(range(node_start, node_start + num_fc2_nodes))
        node_start = node_start + num_fc2_nodes
    
    graph_spec['head_ids'] = split_head_indices(list(range(block_layer_node_num[1])), head)

    graph_spec['head_edge_id'] = list(range(edge_start, edge_start + dim * num_cls))
    graph_spec['head_node_id'] = list(range(node_start, node_start + num_cls))

    return graph_spec