"""
Showing how to use the model with some time series data.

NB! This is not a full training loop. You have to write the training loop yourself. 

I.e. this code is just a starting point to show you how to initialize the model and provide its inputs

If you do not know how to train a PyTorch model, it is too soon for you to dive into transformers imo :) 

You're better off starting off with some simpler architectures, e.g. a simple feed forward network, in order to learn the basics
"""

import torch
import torch.nn.functional as F
import argparse
import datetime
import time
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = '1'
import math
import pandas as pd
import util.dataset as ds
import util.utils as utils
import layer.AGE as tst
import pickle

# from torch_geometric.nn import GATv2Conv
from layer.AGE import GATAE, ConsistentLayer, GAT
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
# from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, get_linear_schedule_with_warmup
# from util.dataset import StandardScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


torch.manual_seed(42)
np.random.seed(42)


def train(model_spatial, optimizer, spatial_loader, scheduler, stack_num, input_size):
    model_spatial.train()
    start_time = time.time()
    # total_loss = 0.

    # for spatial_batch in spatial_loader:
    spatial_batch = spatial_loader
    optimizer.zero_grad()
    # A_pred, z, x_ = model_spatial(spatial_batch.x.to(device), spatial_batch.edge_index.to(device))
    A_pred, z, x_ = model_spatial(spatial_batch.x.to(device), spatial_batch.adj.to(device), spatial_batch.M1, spatial_batch.M2)
    
    # structure reconstruction loss
    re_loss = F.binary_cross_entropy(A_pred.view(-1), spatial_batch.adj_label.view(-1))
    x_loss = F.mse_loss(x_, spatial_batch.x.to(device))

    total_loss = args.lamda*re_loss + x_loss
    
    total_loss.backward()
    optimizer.step()

    ave_batch_loss = total_loss / 1
    elapsed = time.time() - start_time
    print('| epoch {:3d} | lr {:02.8f} | {:5.2f} ms | '
            'step loss {:5.5f} | ppl {:8.2f}'.format(
            epoch, scheduler.get_last_lr()[0], # get_lr()
            elapsed, ave_batch_loss, math.exp(ave_batch_loss))) # math.log(cur_loss)
    
    # total_loss = 0
    start_time = time.time()
    scheduler.step()

    return z

# def test(data, z):
#     pos_score = torch.sigmoid((z[data.edge_index[0]] * z[data.edge_index[1]]).sum(dim=1))
#     neg_edge_index = negative_sampling(data.edge_index, num_nodes=data.num_nodes)
#     neg_score = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))

#     y_true = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
#     y_score = torch.cat([pos_score, neg_score]).cpu().numpy()
#     auc = roc_auc_score(y_true, y_score)
#     ap = average_precision_score(y_true, y_score)

#     return auc, ap

# def predict_link(z, node_a, node_b):
    # score = torch.sigmoid((z[node_a] * z[node_b]).sum())

    # return score.item()

def scale_wozeros(arr, scaler):
    # Replace zeros with NaN
    arr = np.where(arr==0, np.nan, arr)
    # Perform fit_transform on non-NaN values
    scaled_arr = scaler.fit_transform(arr)
    # Replace NaNs back with zeros
    scaled_arr = np.where(np.isnan(scaled_arr), 0, scaled_arr)
    return scaled_arr


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training"
    )
    argparser.add_argument("--data_file", type=str, default='node_truncated_TII',
                           help="file name wo/ suffix")
    argparser.add_argument("--projection_map_file", type=str, default='projection_map_truncated_TII',
                           help="file name wo/ suffix")
    argparser.add_argument("--edgeIdx_file", type=str, default='edge_truncated_TII',
                           help="file name wo/ suffix")
    argparser.add_argument("--SCALER", type=bool, default=True)
    argparser.add_argument("--is_undirected", type=bool, default=False)
    argparser.add_argument("--num_epochs", type=int, default=2000)
    argparser.add_argument("--batch_size", type=int, default=4)
    argparser.add_argument("--hid_dim", type=int, default=512)
    argparser.add_argument("--out_dim", type=int, default=256)
    argparser.add_argument("--n_heads", type=int, default=1)
    argparser.add_argument("--lamda", type=float, default=0.1)
    argparser.add_argument("--node_col", type=str, default="Node")
    argparser.add_argument("--sort_col", type=str, default="Node Index")
    argparser.add_argument("--label_col", type=str, default="Node Label")
    argparser.add_argument("--exogenous_vars", type=str, default="Time,Cost",
                           help="attribute list without consistency consideration")
    argparser.add_argument("--target_col_name", type=str, default="Trust,Flexibility,Risk,Service,Sustainability,EconomyIndex,PoliticalIndex,PortQuality,AssetUtilizationRate,OrderRating",
                           help="split by comma, should contain strings. Attributes for consistent measurement.")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = torch.device("cuda:%d" % args.gpu)
    else:
        device = torch.device("cpu")

    # Define input variables
    exogenous_vars = args.exogenous_vars.split(',')
    input_variables = exogenous_vars + args.target_col_name.split(',')
    stack_num = len(exogenous_vars)

    # Read data
    # Input x
    # (nodes, features)
    data = utils.read_data(file_name=args.data_file, node_col_name=args.node_col, sort_col=args.sort_col)

    # looks like normalizing input values curtial for the model
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = StandardScaler()
    # Recover the original values
    # original_data = scaler.inverse_transform(scaled_data)
    map_series = data[input_variables].values
    labels = data[args.label_col].values

    # dic for label wise feature projection, e.g., OrderedDict([(0, 3), (1, 2))])
    dic = utils.read_projection_map(file_name=args.projection_map_file) # d = OrderedDict(zip(df.values[:,0], df.values[:,1]))
    # 0 avoid the impact of -1 values for the scaler func.
    series = np.zeros((len(map_series), sum(dic.values()))) # -len(input_list)+1 remove other exogenous_vars in each loop
    # series = np.full((len(map_series), sum(dic.values())), -1.) # -1 denotes the absence feature of each node
    for i in range(len(series)):
        given_index = labels[i]
        index = utils.index_for_feature_projection(dic, given_index)
        series[i][index:index+dic[given_index]] = map_series[i][map_series[i] != -1]

    if args.SCALER:
        # scaler = StandardScaler(mean=torch.FloatTensor(series).to(device).mean(axis=0), std=torch.FloatTensor(series).to(device).std(axis=0))
        amplitude = scaler.fit_transform(series)
        # amplitude = normalize(series)
        # amplitude = scale_wozeros(series, scaler)
        amplitude = torch.FloatTensor(amplitude).to(device)
    else:
        amplitude = torch.FloatTensor(series).to(device)

    # # concate onehot label tensor
    # labels_onehot = torch.FloatTensor(utils.encode_onehot(labels)).to(device)
    # amplitude = torch.cat((amplitude, labels_onehot), dim=-1)

    # prepare graph data for link prediction
    input_size = amplitude.size(-1)
    
    # edge index including the start and end nodes.
    edge_data = utils.read_edgeIdx(file_name=args.edgeIdx_file)
    edge_list_ori = edge_data[["startIdx","endIdx"]].values.T
    edge_list=torch.tensor(edge_list_ori, dtype=torch.long)
    if args.is_undirected:
        edge_list = torch.cat([edge_list, torch.flip(edge_list, [0])], dim=-1) # is_undirected
    # labels = np.random.randint(0, 3, 25) # num_classes
    # node_type = torch.tensor(data['Node Label'].values, dtype=torch.long)
    # edge_type = torch.tensor(edge_data['Relation Label'].values, dtype=torch.long)

    # Create a PyTorch Geometric Data object
    output_data = Data(x=amplitude,
                edge_index=edge_list)
    # output_data.to_heterogeneous()
    
    # Split edges using RandomLinkSplit
    split = RandomLinkSplit(num_val=0.0, num_test=0.3, # revise line 214 to be "force_undirected=self.is_undirected"
                            add_negative_train_samples=True, neg_sampling_ratio=1,
                            split_labels=True,
                            is_undirected=args.is_undirected) # edge_label_index[:, train_data.edge_label==1]
    train_data, val_data, test_data = split(output_data)
    if args.is_undirected:
        train_edges = torch.cat([train_data.pos_edge_label_index, torch.flip(train_data.pos_edge_label_index, [0])], dim=-1) # is_undirected
        train_edges_neg=torch.cat([train_data.neg_edge_label_index, torch.flip(train_data.neg_edge_label_index, [0])], dim=-1)
        test_edges=torch.cat([test_data.pos_edge_label_index, torch.flip(test_data.pos_edge_label_index, [0])], dim=-1)
        test_edges_neg=torch.cat([test_data.neg_edge_label_index, torch.flip(test_data.neg_edge_label_index, [0])], dim=-1)
    else:
        train_edges=train_data.pos_edge_label_index
        train_edges_neg=train_data.neg_edge_label_index
        test_edges=test_data.pos_edge_label_index
        test_edges_neg=test_data.neg_edge_label_index
    
    adj, adj_label, adj_sp = utils.generate_spadj_from_edgeidx(train_edges.cpu(), output_data.num_nodes)
    adj, adj_label, adj_sp = adj.to(device), adj_label.to(device), adj_sp.to(device)
    M1 = utils.get_M(adj,1).to(device)
    M2 = utils.get_M(adj,2).to(device)

    # spatial_loader = train_data
    spatial_loader = Data(x=output_data.x,
        adj=adj,
        adj_label=adj_label,
        M1=M1,
        M2=M2,
        edge_index=test_edges,
        neg_edge_index=test_edges_neg) ### to_heterogeneous
    # spatial_loader = Data(x=output_data.x,
    #         adj_label=adj_label,
    #         edge_index=train_edges,
    #         neg_edge_index=train_edges_neg) ### to_heterogeneous

    # graph loader
    # spatial_loader = NeighborLoader(
    #         train_data,
    #         # Sample 30 neighbors for each node for 2 iterations
    #         num_neighbors=[1] * 2,
    #         # Use a batch size of 128 for sampling training nodes
    #         batch_size=args.batch_size,
    #         input_nodes=torch.unique(train_data.edge_index.flatten(start_dim=0))
    #     )
    
    # model_spatial = GATAE(in_channels=input_size, hidden_channels=args.hid_dim, out_channels=args.out_dim, nheads=args.n_heads).to(device)
    model_spatial = GAT(num_features=input_size, hidden_size=args.hid_dim, embedding_size=args.out_dim).to(device)
    # consitent_layer = ConsistentLayer(in_channels=input_size*stack_num, out_channels=input_size).to(device)

    optimizer = torch.optim.AdamW(model_spatial.parameters(), lr=0.005)

    # Define the warm-up schedule
    # total_steps = len(training_time_data) * num_epochs
    # Create the scheduler
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=num_epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.num_epochs//10, gamma=0.95) # args.num_epochs//10


    for epoch in range(args.num_epochs+1):
        output = train(model_spatial, optimizer, spatial_loader, scheduler, stack_num, input_size)

        if epoch == args.num_epochs:
            model_spatial.eval()
            # _, z, x_ = model_spatial(output_data.x.to(device), train_edges.to(device))
            _, z, x_ = model_spatial(output_data.x.to(device), adj.to(device), M1, M2)
            node_embeddings_pos = z[train_edges[0]] * z[train_edges[1]]
            node_embeddings_neg = z[train_edges_neg[0]] * z[train_edges_neg[1]]
            node_embeddings = torch.cat((node_embeddings_pos, node_embeddings_neg), dim=0).detach().cpu().numpy()
            labels_train = torch.cat([torch.ones(node_embeddings_pos.shape[0]), torch.zeros(node_embeddings_neg.shape[0])]).cpu().numpy()
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(node_embeddings, labels_train)

            # model_spatial.eval()
            node_embeddings_pos = z[test_edges[0]] * z[test_edges[1]]
            node_embeddings_neg = z[test_edges_neg[0]] * z[test_edges_neg[1]]
            node_embeddings = torch.cat((node_embeddings_pos, node_embeddings_neg), dim=0).detach().cpu().numpy()
            y_true = torch.cat([torch.ones(node_embeddings_pos.shape[0]), torch.zeros(node_embeddings_neg.shape[0])]).cpu().numpy()
            y_score = rf.predict(node_embeddings)
            
            roc_auc = roc_auc_score(y_true, y_score)
            pr_auc = average_precision_score(y_true, y_score)
            precision = precision_score(y_true, y_score)
            recall = recall_score(y_true, y_score)
            f1 = f1_score(y_true, y_score)

            print('| precision {:3f} | recall {:3f} | f1 {:3f} | AUC-ROC {:3f} | AUC-PR {:3f}'.format(precision, recall, f1, roc_auc, pr_auc))

            # Save
            with open('model/rfClassifier_GAT.pkl', 'wb') as f:
                pickle.dump(rf, f)

        if (epoch) % 10 == 0:
            # Save the model
            torch.save(model_spatial.state_dict(), 'model/modelSpatial_GAT.pth')
            # model.load_state_dict(torch.load('model.pth'))

            # # evaluation
            # auc, ap = test(model, test_data, device)
            # print(f'Epoch: {epoch}, AUC: {auc:.4f}, AP: {ap:.4f}')
