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

from sklearn.cluster import KMeans, SpectralClustering
from munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear

from sklearn import metrics
torch.manual_seed(42)
np.random.seed(42)


def update_similarity(z, upper_threshold, lower_treshold, pos_num, neg_num):
    f_adj = np.matmul(z, np.transpose(z))
    cosine = f_adj
    cosine = cosine.reshape([-1,])
    pos_num = round(upper_threshold * len(cosine))
    neg_num = round((1-lower_treshold) * len(cosine))
    
    # cosine大,相似性大
    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]
    
    return np.array(pos_inds), np.array(neg_inds)

def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
    upth = upper_threshold + up_eta
    lowth = lower_treshold + low_eta
    return upth, lowth

def loss_function(adj_preds, adj_labels, n_nodes):
    cost = 0.
    cost += F.binary_cross_entropy_with_logits(adj_preds, adj_labels)
    
    return cost

def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def scale_wozeros(arr, scaler):
    # Replace zeros with NaN
    arr = np.where(arr==0, np.nan, arr)
    # Perform fit_transform on non-NaN values
    scaled_arr = scaler.fit_transform(arr)
    # Replace NaNs back with zeros
    scaled_arr = np.where(np.isnan(scaled_arr), 0, scaled_arr)
    return scaled_arr

def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method="arithmetic")
    ari = ari_score(y_true, y_pred)
    print(f"epoch {epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")
    return acc, nmi, ari, f1

# similar to https://github.com/karenlatong/AGC-master/blob/master/metrics.py
def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print("error")
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average="macro")
    precision_macro = metrics.precision_score(y_true, new_predict, average="macro")
    recall_macro = metrics.recall_score(y_true, new_predict, average="macro")
    f1_micro = metrics.f1_score(y_true, new_predict, average="micro")
    precision_micro = metrics.precision_score(y_true, new_predict, average="micro")
    recall_micro = metrics.recall_score(y_true, new_predict, average="micro")
    return acc, f1_macro


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
    argparser.add_argument("--hid_dim", type=int, default=256)
    argparser.add_argument("--out_dim", type=int, default=16)
    argparser.add_argument("--n_heads", type=int, default=1)
    argparser.add_argument("--lamda", type=float, default=0.1)
    argparser.add_argument('--upth_st', type=float, default=0.11, help='Upper Threshold start.')
    argparser.add_argument('--lowth_st', type=float, default=0.1, help='Lower Threshold start.')
    argparser.add_argument('--upth_ed', type=float, default=0.001, help='Upper Threshold end.')
    argparser.add_argument('--lowth_ed', type=float, default=0.5, help='Lower Threshold end.')
    argparser.add_argument('--bs', type=int, default=1000, help='Batchsize.')
    argparser.add_argument('--update_interval', default=5, type=int)  # [1,3,5]
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
    split = RandomLinkSplit(num_val=0.05, num_test=0.3, # revise line 214 to be "force_undirected=self.is_undirected"
                            add_negative_train_samples=True, neg_sampling_ratio=1,
                            split_labels=True,
                            is_undirected=args.is_undirected) # edge_label_index[:, train_data.edge_label==1]
    train_data, val_data, test_data = split(output_data)
    if args.is_undirected:
        train_edges = torch.cat([train_data.pos_edge_label_index, torch.flip(train_data.pos_edge_label_index, [0])], dim=-1) # is_undirected
        train_edges_neg=torch.cat([train_data.neg_edge_label_index, torch.flip(train_data.neg_edge_label_index, [0])], dim=-1)
        val_edges=torch.cat([val_data.pos_edge_label_index, torch.flip(val_data.pos_edge_label_index, [0])], dim=-1)
        val_edges_neg=torch.cat([val_data.neg_edge_label_index, torch.flip(val_data.neg_edge_label_index, [0])], dim=-1)
        test_edges=torch.cat([test_data.pos_edge_label_index, torch.flip(test_data.pos_edge_label_index, [0])], dim=-1)
        test_edges_neg=torch.cat([test_data.neg_edge_label_index, torch.flip(test_data.neg_edge_label_index, [0])], dim=-1)
    else:
        train_edges=train_data.pos_edge_label_index
        train_edges_neg=train_data.neg_edge_label_index
        val_edges=val_data.pos_edge_label_index
        val_edges_neg=val_data.neg_edge_label_index
        test_edges=test_data.pos_edge_label_index
        test_edges_neg=test_data.neg_edge_label_index
    
    adj_orig, _, _ = utils.generate_spadj_from_edgeidx(output_data.edge_index.cpu(), output_data.num_nodes)
    adj, adj_label, adj_sp = utils.generate_spadj_from_edgeidx(train_edges.cpu(), output_data.num_nodes)
    adj, adj_label, adj_sp = adj.to(device), adj_label.to(device), adj_sp.to(device)
    M1 = utils.get_M(adj,1).to(device)
    M2 = utils.get_M(adj,2).to(device)

    # spatial_loader = train_data
    spatial_loader = Data(x=output_data.x,
            adj_label=adj_label,
            adj=adj,
            M1=M1,
            M2=M2) ### to_heterogeneous
    # graph loader
    # spatial_loader = NeighborLoader(
    #         train_data,
    #         # Sample 30 neighbors for each node for 2 iterations
    #         num_neighbors=[1] * 2,
    #         # Use a batch size of 128 for sampling training nodes
    #         batch_size=args.batch_size,
    #         input_nodes=torch.unique(train_data.edge_index.flatten(start_dim=0))
    #     )
    
    model_spatial = GAT(num_features=input_size, hidden_size=args.hid_dim, embedding_size=args.out_dim).to(device)
    # consitent_layer = ConsistentLayer(in_channels=input_size*stack_num, out_channels=input_size).to(device)

    optimizer = torch.optim.AdamW(model_spatial.parameters(), lr=0.005)

    # Define the warm-up schedule
    # total_steps = len(training_time_data) * num_epochs
    # Create the scheduler
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=num_epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.num_epochs//10, gamma=0.95) # args.num_epochs//10


    # Training Sample selection
    n_nodes, _ = output_data.x.shape
    pos_num = len(adj_sp.coalesce().indices())
    neg_num = n_nodes*n_nodes-pos_num

    up_eta = (args.upth_ed - args.upth_st) / (args.num_epochs/args.update_interval)
    low_eta = (args.lowth_ed - args.lowth_st) / (args.num_epochs/args.update_interval)

    with torch.no_grad():
        _, z, _ = model_spatial(output_data.x.to(device), adj.to(device), M1, M2)

    pos_inds, neg_inds = update_similarity(z.cpu().data.numpy(), args.upth_st, args.lowth_st, pos_num, neg_num)
    upth, lowth = update_threshold(args.upth_st, args.lowth_st, up_eta, low_eta)

    bs = min(args.bs, len(pos_inds))
    length = len(pos_inds)
    
    pos_inds_cuda = torch.LongTensor(pos_inds).to(device)

    best_acc, best_nmi, best_ari, best_up, best_low, best_lp = 0, 0, 0, 0, 0, 0

    for epoch in range(args.num_epochs):
        st, ed = 0, length
        batch_num = 0
        length = len(pos_inds)

        model_spatial.train()
        start_time = time.time()

        while ( ed <= length ):
            # for spatial_batch in spatial_loader:
            spatial_batch = spatial_loader
            optimizer.zero_grad()
            A_pred, z, x_ = model_spatial(output_data.x.to(device), adj.to(device), M1, M2)
            
            # node similarity decoder
            sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=ed-st)).to(device)
            sampled_inds = torch.cat((pos_inds_cuda[st:ed], sampled_neg), 0)
            # node x and node y
            zind = torch.div(sampled_inds, n_nodes).to(torch.int32)
            # zind = sampled_inds // n_nodes
            ztind = sampled_inds % n_nodes

            z_sample = torch.index_select(z, 0, zind)
            zt_sample = torch.index_select(z, 0, ztind)

            batch_label = torch.cat((torch.ones(ed-st), torch.zeros(ed-st))).to(device)
            batch_pred = model_spatial.dcs(z_sample, zt_sample)
            
            re_loss = loss_function(batch_pred, batch_label, n_nodes=ed-st)
            # structure reconstruction loss
            ra_loss = F.binary_cross_entropy(A_pred.view(-1), spatial_batch.adj_label.view(-1))
            rn_loss = F.mse_loss(x_, spatial_batch.x.to(device))

            total_loss = 0.1*re_loss + 1*ra_loss + 1*rn_loss
            
            total_loss.backward()
            optimizer.step()

            st = ed
            batch_num += 1
            if ed < length and ed + bs >= length:
                ed += length - ed
            else:
                ed += bs

            ave_batch_loss = total_loss / 1
            elapsed = time.time() - start_time
            print('| epoch {:3d} | lr {:02.8f} | {:5.2f} ms | '
                    'step loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, scheduler.get_last_lr()[0], # get_lr()
                    elapsed, ave_batch_loss, math.exp(ave_batch_loss))) # math.log(cur_loss)
            
            # total_loss = 0
            start_time = time.time()
            # scheduler.step()

            model_spatial.eval()
            _, mu, _ = model_spatial(output_data.x.to(device), adj.to(device), M1, M2)
            hidden_emb = mu.cpu().data.numpy()
            val_auc, val_ap = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_neg)
            if val_auc + val_ap >= best_lp:
                best_lp = val_auc + val_ap
                best_emb = hidden_emb

        if epoch % args.update_interval == 0 and val_auc + val_ap == best_lp:

            upth, lowth = update_threshold(upth, lowth, up_eta, low_eta)
            pos_inds, neg_inds = update_similarity(hidden_emb, upth, lowth, pos_num, neg_num)
            bs = min(args.bs, len(pos_inds))
            pos_inds_cuda = torch.LongTensor(pos_inds).to(device)


            # kmeansEval = KMeans(n_clusters=labels.max()+1, n_init=20).fit(hidden_emb)
            # acc, nmi, ari, f1 = eva(labels, kmeansEval.labels_, epoch)

            # if acc >= best_acc and epoch > (args.num_epochs // 2):
            #     best_acc, best_nmi, best_ari, best_f1 = acc, nmi, ari, f1
            #     best_up = upth
            #     best_low = lowth
            #     torch.save(model_spatial.state_dict(), 'model/rfClassifier_GAT.pkl')

        # if epoch == args.num_epochs:
        #     model_spatial.eval()
        #     _, z, x_ = model_spatial(output_data.x.to(device), train_edges.to(device))
        #     node_embeddings_pos = z[train_edges[0]] * z[train_edges[1]]
        #     node_embeddings_neg = z[train_edges_neg[0]] * z[train_edges_neg[1]]
        #     node_embeddings = torch.cat((node_embeddings_pos, node_embeddings_neg), dim=0).detach().cpu().numpy()
        #     labels_train = torch.cat([torch.ones(node_embeddings_pos.shape[0]), torch.zeros(node_embeddings_neg.shape[0])]).cpu().numpy()
        #     rf = RandomForestClassifier(n_estimators=100, random_state=42)
        #     rf.fit(node_embeddings, labels_train)

        #     # model_spatial.eval()
        #     node_embeddings_pos = z[test_edges[0]] * z[test_edges[1]]
        #     node_embeddings_neg = z[test_edges_neg[0]] * z[test_edges_neg[1]]
        #     node_embeddings = torch.cat((node_embeddings_pos, node_embeddings_neg), dim=0).detach().cpu().numpy()
        #     y_true = torch.cat([torch.ones(node_embeddings_pos.shape[0]), torch.zeros(node_embeddings_neg.shape[0])]).cpu().numpy()
        #     y_score = rf.predict(node_embeddings)
            
        #     roc_auc = roc_auc_score(y_true, y_score)
        #     pr_auc = average_precision_score(y_true, y_score)
        #     precision = precision_score(y_true, y_score)
        #     recall = recall_score(y_true, y_score)
        #     f1 = f1_score(y_true, y_score)

        #     print('| precision {:3f} | recall {:3f} | f1 {:3f} | AUC-ROC {:3f} | AUC-PR {:3f}'.format(precision, recall, f1, roc_auc, pr_auc))

        #     # Save
        #     with open('model/rfClassifier_GAT.pkl', 'wb') as f:
        #         pickle.dump(rf, f)

        if (epoch) % 10 == 0:
            # Save the model
            torch.save(model_spatial.state_dict(), 'model/modelSpatial_GAT.pth')
            # model.load_state_dict(torch.load('model.pth'))

            # # evaluation
            # auc, ap = test(model, test_data, device)
            # print(f'Epoch: {epoch}, AUC: {auc:.4f}, AP: {ap:.4f}')
    
    auc_score, ap_score = get_roc_score(best_emb, adj_orig, test_edges, test_edges_neg)
    print(f"auc {auc_score:.4f}, ap {ap_score:.4f}")
    # print(f"best up {best_up}, best low {best_low}:acc {best_acc:.4f}, nmi {best_nmi:.4f}, ari {best_ari:.4f}, f1 {best_f1:.4f}")
