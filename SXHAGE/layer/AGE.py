
import torch
import torch.nn as nn 
import torch.nn.functional as F
import math
import numpy as np

from torch import nn, Tensor
from torch_geometric.nn import GATConv, GATv2Conv
# from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
# from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn import SAGEConv
from .sagan_models import Self_Attn, Attn
from sklearn.preprocessing import normalize
from layer.positional_encoder import PositionalEncoder


class GATAE(torch.nn.Module):
    """
    Two GATv2 layer, https://github.com/tech-srl/how_attentive_are_gats
    """
    def __init__(self, in_channels, hidden_channels, out_channels, nheads=1, dropout=0.2, stack_num=1, concat=True, GATv2=False):
        super(GATAE, self).__init__()
        if GATv2:
            self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=nheads, dropout=dropout, concat=concat)
            self.conv2 = GATv2Conv(nheads * hidden_channels, out_channels, heads=1, concat=False, dropout=dropout)
        else:
            self.conv1 = GATConv(in_channels, hidden_channels, heads=nheads, dropout=dropout, concat=concat)
            self.conv2 = GATConv(nheads * hidden_channels, out_channels, heads=1, concat=False, dropout=dropout)
        # self.conv3 = GATConv(out_channels, hidden_channels, heads=nheads, dropout=dropout, concat=concat)
        # self.conv4 = GATConv(nheads * hidden_channels, in_channels, heads=1, concat=False, dropout=dropout)
        self.linear = nn.Linear(out_channels, in_channels*stack_num)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.1)
        self.dcs = SampleDecoder(act=lambda x: x)

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = F.elu(self.conv1(x, edge_index))
        h = self.conv1(x, edge_index)
        h = self.conv2(h, edge_index)
        # x_ = self.conv3(x, edge_index)
        # x_ = self.conv4(x_, edge_index)
        z = F.normalize(h, p=2, dim=1)  # Apply L2 normalization
        # x_ = self.dropout(self.elu(self.linear(z)))
        x_ = self.elu(self.linear(z))
        A_pred = self.dot_product_decode(z)
        return A_pred, z, x_

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred


class ConsistentLayer(nn.Module):
    def __init__(self, in_channels, out_channels, rec_channels, stack_num=1):
        super(ConsistentLayer, self).__init__()
        self.self_attention = Self_Attn(in_channels*stack_num)
        self.attention = Attn(in_channels)
        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=in_channels*stack_num,
            dropout=0.1
            )
        self.layer_norm = nn.LayerNorm(in_channels*stack_num)
        self.linear = nn.Linear(in_channels*stack_num, out_channels)
        self.linear_ = nn.Linear(out_channels, rec_channels*stack_num)

        self.elu = nn.ELU()
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, attr_output_list, use_self_attention=True, use_attention=False):
        if use_attention:
            attr_outputs = torch.stack(attr_output_list,dim=1)
            attr_outputs = self.attention(attr_outputs)
            # attr_outputs = self.layer_norm(attr_outputs)
            # attr_outputs = F.normalize(attr_outputs, p=2, dim=1)
            z = self.leakyrelu(self.linear(attr_outputs))
        else:
            attr_outputs = torch.cat(attr_output_list, dim=-1)
            if use_self_attention:
                attr_outputs = self.positional_encoding_layer(attr_outputs)
                attr_outputs = self.self_attention(attr_outputs)
                attr_outputs = self.layer_norm(attr_outputs)
            z = self.leakyrelu(self.linear(attr_outputs))
            # z = self.dropout(self.leakyrelu(self.linear(attr_outputs)))

        x_ = self.elu(self.linear_(z))
        # x_ = self.dropout(self.elu(self.linear_(z)))
        A_pred = self.dot_product_decode(z)
        return A_pred, z, x_
    
    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred


class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.sage_conv1 = SAGEConv(in_channels, hidden_channels)
        self.sage_conv2 = SAGEConv(hidden_channels, out_channels)
        self.linear = nn.Linear(out_channels, in_channels)
        self.elu = nn.ELU()

    def forward(self, x, edge_index):
        h = self.sage_conv1(x, edge_index)
        h = self.sage_conv2(h, edge_index)
        z = F.normalize(h, p=2, dim=1)  # Apply L2 normalization
        x_ = self.elu(self.linear(z))
        # x_ = F.dropout(self.elu(self.linear(z)), p=0.2, training=self.training)
        A_pred = self.dot_product_decode(z)
        return A_pred, z, x_ #F.log_softmax(x, dim=1)
    
    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
    

class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # self.bias = nn.Parameter(torch.zeros(size=(1, out_features)))
        # nn.init.xavier_uniform_(self.bias.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=nn.init.calculate_gain('leaky_relu', 0.2))

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=nn.init.calculate_gain('leaky_relu', 0.2))

        self.a = nn.Parameter(torch.zeros(size=(out_features, 3327)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()

        # self.GCNConv = GCNConv(in_features,out_features)

        # self.GATv2Conv = GATv2Conv(in_features,out_features)

        # self.ClusterGCNConv = ClusterGCNConv(in_features,out_features)

        self.linear = torch.nn.Linear(in_features,out_features,bias=True)

        self.beta = nn.Parameter(torch.Tensor(1).uniform_(0, 1), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros(size=(3327, 3327)))
        # nn.init.xavier_uniform_(self.beta.data, gain=nn.init.calculate_gain('leaky_relu', 0.2))
        # self.beta = Variable(torch.zeros(1), requires_grad=True).cuda()


    def forward(self, input, adj, M, concat=True):
        h = self.linear(input)
        # h = self.elu(h)  # (N,N) self.tanh(h)

        # h_prime = self.GATv2Conv(input, edge)
        # h_prime = self.GCNConv(input, edge)
        # h_prime = self.ClusterGCNConv(input, edge)

        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        # masked = torch.mm(h_att, self.a)  # (N,N)
        # attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)  # (N,N)
        # attn_dense = torch.mul(attn_dense, M)

        # NaN grad bug fixed at pytorch 0.3. Release note:
        #     when torch.norm returned 0.0, the gradient was NaN.
        #     We now use the subgradient at 0.0, so the gradient is 0.0.`
        norm2_self = torch.norm(attn_for_self, 2, 1).view(-1, 1)
        norm2_neighs = torch.norm(attn_for_neighs, 2, 1).view(-1, 1)

        # add a minor constant (1e-7) to denominator to prevent division by
        # zero error
        cos = torch.div(torch.mm(attn_for_self, attn_for_neighs.t()), torch.mm(norm2_self, norm2_neighs.t()) + 1e-7)
        mask = self.leakyrelu(cos)  # (N,N)
        # cos = torch.matmul(cos, M)
        masked = self.beta * mask

        # neighborhood masking (inspired by this repo:
        # https://github.com/danielegrattarola/keras-gat)
        # mask = (1. - adj) * -1e9
        # masked = cos + mask


        # propagation matrix
        # attention = F.softmax(masked, dim=1)

        zero_vec = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, masked, zero_vec)
        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention, h)

        if concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
    
class GAT(nn.Module):
    """
    Two GAT layer, https://github.com/Tiger101010/DAEGC
    """
    def __init__(self, num_features, hidden_size, embedding_size, alpha=0.2):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)
        self.deconv1 = GATLayer(embedding_size, hidden_size, alpha)
        self.deconv2 = GATLayer(hidden_size, num_features, alpha)
        self.dcs = SampleDecoder(act=lambda x: x)
        # self.linear = nn.Linear(embedding_size, num_features)
        # self.elu = nn.ELU()
        # self.dropout = nn.Dropout(p=0.1)

        # node attribute reconstruction
        self.cls_y = torch.nn.Linear(embedding_size, num_features)

    def forward(self, x, adj, M1=None, M2=None):
        h = self.conv1(x, adj, M1)
        h = self.conv2(h, adj, M2)
        z = F.normalize(h, p=2, dim=1)
        x_ = self.deconv1(z, adj, M1)
        x_ = self.deconv2(x_, adj, M2)
        # x_ = self.dropout(self.elu(self.linear(z)))
        # x_ = self.elu(self.linear(z))
        A_pred = self.dot_product_decode(z)
        return A_pred, z, x_

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

class SampleDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid):
        super(SampleDecoder, self).__init__()
        self.act = act

    def forward(self, zx, zy):
        # torch.nn.CosineSimilarity
        sim = (zx * zy).sum(1)
        sim = self.act(sim)
    
        return sim
    

class HeterogeneousGAT(nn.Module): # heterogeneous relations, but remains for revision
    def __init__(self, num_node_features, num_relations, hidden_dim, num_heads):
        super(HeterogeneousGAT, self).__init__()
        self.num_relations = num_relations
        self.attentions = nn.ModuleList()
        for _ in range(num_relations):
            self.attentions.append(GATConv(num_node_features, hidden_dim, heads=num_heads))

    def forward(self, x, edge_index, relation_ids):
        # x: node features (shape: [num_nodes, num_node_features])
        # edge_index: edge indices (shape: [2, num_edges])
        # relation_ids: relation type identifiers for each edge (shape: [num_edges])

        # Apply GAT convolution for each relation type
        relation_representations = []
        for i in range(self.num_relations):
            attention = self.attentions[i]
            relation_mask = (relation_ids == i)
            edge_index_i = edge_index[:, relation_mask]
            x_i = x.clone()
            x_i = attention(x_i, edge_index_i)
            relation_representations.append(x_i)

        # Concatenate relation-specific representations
        x_out = torch.cat(relation_representations, dim=1)

        return x_out    