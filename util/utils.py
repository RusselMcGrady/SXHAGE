import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

from collections import OrderedDict
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable, Tuple
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling
from sklearn.preprocessing import normalize


def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:

        dim1: int, for both src and tgt masking, this must be target sequence
              length

        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 


    Return:

        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


def get_indices_input_target(input_len, step_size, slice_size):
    """
    Produce all the start and end index positions of all sub-sequences.
    The indices will be used to split the data into sub-sequences on which 
    the models will be trained. 

    Returns a tuple with four elements:
    1) The index position of the first element to be included in the input sequence
    2) The index position of the last element to be included in the input sequence
    3) The index position of the first element to be included in the target sequence
    4) The index position of the last element to be included in the target sequence

    
    Args:
        num_obs (int): Number of observations in the entire dataset for which
                        indices must be generated.

        input_len (int): Length of the input sequence (a sub-sequence of 
                            of the entire data sequence)

        step_size (int): Size of each step as the data sequence is traversed.
                            If 1, the first sub-sequence will be indices 0-input_len, 
                            and the next will be 1-input_len.

        forecast_horizon (int): How many index positions is the target away from
                                the last index position of the input sequence?
                                If forecast_horizon=1, and the input sequence
                                is data[0:10], the target will be data[11:taget_len].

        target_len (int): Length of the target / output sequence.

        slice_size (int): num of series slice for each node.
    """
    input_len = round(input_len) # just a precaution
    start_position = 0
    stop_position = slice_size
    
    subseq_first_idx = start_position
    subseq_last_idx = start_position + input_len
    # target_first_idx = subseq_last_idx + forecast_horizon
    # target_last_idx = target_first_idx + target_len 
    # print("target_last_idx is {}".format(target_last_idx))
    print("stop_position is {}".format(stop_position))
    indices = []
    while subseq_last_idx <= stop_position:
        # indices.append((subseq_first_idx, subseq_last_idx, target_first_idx, target_last_idx))
        indices.append((subseq_first_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_last_idx += step_size
        # target_first_idx = subseq_last_idx + forecast_horizon
        # target_last_idx = target_first_idx + target_len

    return indices

    # todo: cannot slice series for different nodes
    # input_len = round(input_len) # just a precaution
    # indices = []
    
    # for i in range(round(num_obs//slice_size)):
    #     sub_indices = []
    #     stop_position = slice_size*(i+1) # for each node
        
    #     # Start the first sub-sequence at index position 0
    #     subseq_first_idx = 0 + i*slice_size
    #     subseq_last_idx = input_len + i*slice_size
    #     # pred_first_idx = subseq_last_idx + forecast_horizon
    #     # pred_last_idx = pred_first_idx + target_len 
    #     # print("target_last_idx is {}".format(target_last_idx))
    #     # print("stop_position is {}".format(stop_position))

    #     while subseq_last_idx <= stop_position:
    #         # indices.append((subseq_first_idx, subseq_last_idx, target_first_idx, target_last_idx))
    #         sub_indices.append((subseq_first_idx, subseq_last_idx))
    #         subseq_first_idx += step_size
    #         subseq_last_idx += step_size
    #         # target_first_idx = subseq_last_idx + forecast_horizon
    #         # target_last_idx = target_first_idx + target_len

    #     indices.append(sub_indices)

    # return indices

def get_indices_entire_sequence(window_size: int, step_size: int, slice_size: int) -> list:
    """
    Produce all the start and end index positions that is needed to produce
    the sub-sequences. 

    Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
    sequence. These tuples should be used to slice the dataset into sub-
    sequences. These sub-sequences should then be passed into a function
    that slices them into input and target sequences. 
    
    Args:
        num_obs (int): Number of observations (time steps) in the entire 
                        dataset for which indices must be generated, e.g. 
                        len(data)

        window_size (int): The desired length of each sub-sequence. Should be
                            (input_sequence_length + target_sequence_length)
                            E.g. if you want the model to consider the past 100
                            time steps in order to predict the future 50 
                            time steps, window_size = 100+50 = 150

        step_size (int): Size of each step as the data sequence is traversed 
                            by the moving window.
                            If 1, the first sub-sequence will be [0:window_size], 
                            and the next will be [1:window_size].

        slice_size (int): num of series slice for each node.

    Return:
        indices: a list of tuples
    """
    stop_position = slice_size
    
    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0
    
    subseq_last_idx = window_size
    
    indices = []
    
    while subseq_last_idx <= stop_position:

        indices.append((subseq_first_idx, subseq_last_idx))
        
        subseq_first_idx += step_size
        
        subseq_last_idx += step_size

    return indices

    # todo: cannot slice series for different nodes
    # indices = []

    # for i in range(round(len(data)//slice_size)):
    #     sub_indices = []
    #     stop_position = slice_size + i*slice_size # 1- because of 0 indexing len(data)-1
        
    #     # Start the first sub-sequence at index position 0
    #     subseq_first_idx = 0 + i*slice_size
        
    #     subseq_last_idx = window_size + i*slice_size
        
    #     while subseq_last_idx <= stop_position:

    #         sub_indices.append((subseq_first_idx, subseq_last_idx))
            
    #         subseq_first_idx += step_size
            
    #         subseq_last_idx += step_size

    #     indices.append(sub_indices)

    # return indices


def read_data(data_dir: Union[str, Path] = "data", file_name: str="dfs_merged_upload",
    node_col_name: str="Node", sort_col: str="Node Index") -> pd.DataFrame:
    """
    Read data from csv file and return pd.Dataframe object

    Args:

        data_dir: str or Path object specifying the path to the directory 
                  containing the data

        target_col_name: str, the name of the column containing the target variable

        timestamp_col_name: str, the name of the column or named index 
                            containing the timestamps
    """

    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # Read csv file
    csv_files = list(data_dir.glob(file_name+".csv"))
    
    if len(csv_files) > 1:
        # raise ValueError("data_dir contains more than 1 csv file. Must only contain 1")
        pass
    elif len(csv_files) == 0:
         raise ValueError("data_dir must contain at least 1 csv file.")

    data_path = csv_files[0]

    print("Reading file in {}".format(data_path))

    data = pd.read_csv(
        data_path,
        low_memory=False
    )

    # Make sure all "n/e" values have been removed from df. 
    if is_ne_in_df(data):
        raise ValueError("data frame contains 'n/e' values. These must be handled")
    
    data = to_numeric_and_downcast_data(data)

    # Make sure data is in ascending order by timestamp
    data.sort_values(by=[sort_col], inplace=True)

    return data

def is_ne_in_df(df:pd.DataFrame):
    """
    Some raw data files contain cells with "n/e". This function checks whether
    any column in a df contains a cell with "n/e". Returns False if no columns
    contain "n/e", True otherwise
    """
    
    for col in df.columns:

        true_bool = (df[col] == "n/e")

        if any(true_bool):
            return True

    return False

def to_numeric_and_downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes('float').columns
    
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df

def read_projection_map(data_dir: Union[str, Path] = "data", file_name: str="dfs_merged_upload") -> OrderedDict:
    
    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # Read csv file
    csv_files = list(data_dir.glob(file_name+".csv"))
    
    if len(csv_files) == 0:
         raise ValueError("data_dir must contain at least 1 csv file.")

    data_path = csv_files[0]

    print("Reading file in {}".format(data_path))

    df = pd.read_csv(data_path)

    # Create an OrderedDict from the DataFrame:
    d = OrderedDict(zip(df.values[:,0], df.values[:,1]))
    return d

def read_edgeIdx(data_dir: Union[str, Path] = "data", file_name: str="dfs_merged_upload") -> list[list[int]]:
    
    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # Read csv file
    csv_files = list(data_dir.glob(file_name+".csv"))
    
    if len(csv_files) == 0:
         raise ValueError("data_dir must contain at least 1 csv file.")

    data_path = csv_files[0]

    print("Reading file in {}".format(data_path))

    df = pd.read_csv(data_path)

    # # set and write the edge index randomly
    # edge_index = torch.randint(1505, (2, 5000))
    # for i in range(edge_index.size()[1]):
    #     if i<=len(df):
    #         df["startIdx"][i] = edge_index[0][i].numpy()
    #         df["endIdx"][i] = edge_index[1][i].numpy()
    #     else:
    #         # append a new row with index 3
    #         new_row = pd.Series({'startIdx': edge_index[0][i].numpy(), 'endIdx': edge_index[1][i].numpy()})
    #         df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    # df.to_csv(data_path, index=False)
    
    return df


def index_for_feature_projection(d, given_index):
    sum_values = 0
    for _, (key, value) in enumerate(d.items()):
        if key != given_index:
            sum_values += value
        else:
            break
    return sum_values


def generate_adj_from_edgeidx(edge_index: Tensor, num_nodes) -> Tensor:
    """
    generate adj given the edge index
    the edge_index tensor to be in the compressed sparse row (CSR) format, which requires the column indices to be sorted.
    Example edge_index tensor edge_index = torch.tensor([[0, 0, 1, 1, 2, 3, 3, 4], [1, 3, 0, 2, 3, 1, 4, 3]])
    """

    # # Compute number of nodes from edge_index tensor
    # num_nodes = edge_index.max().item() + 1

    # Create a sparse COO tensor with ones at the locations specified by the edge_index tensor
    adj_matrix = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), 
                                        (num_nodes, num_nodes))

    # Convert sparse tensor to dense tensor
    adj_matrix = adj_matrix.to_dense()

    # return adjacency matrix
    return adj_matrix

def generate_spadj_from_edgeidx(edge_index: Tensor, num_nodes) -> Tensor:
    adj_sp = torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.shape[1]), torch.Size([num_nodes, num_nodes])
    )
    adj = adj_sp.to_dense()
    adj_label = adj #.detach().clone()

    adj += torch.eye(num_nodes)
    adj = normalize(adj, norm="l1")
    adj = torch.from_numpy(adj).to(dtype=torch.float)

    return adj, adj_label, adj_sp

def generate_edgeidx_from_adj(adj: Tensor) -> Tensor:
    # obtain the indices of the non-zero elements
    row, col = torch.where(adj != 0)
    # concatenate row and col tensors to obtain edge index
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


# eval for time series forecasting
def plot(output, truth, step, plot_length, loss):
    plot_length = plot_length
    # Define the colors list with a gradient
    # colors = plt.cm.jet(np.linspace(0, 1, output.size()[1]*3))
    print('-'*12)
    print('validation loss : {:5.5f}'.format(loss))
    print('-'*12)

    for i in range(output.size()[1]):
        plt.plot(output[:plot_length,i], color='red')
        plt.plot(truth[:plot_length,i], color='blue')
        plt.plot(output[:plot_length,i]-truth[:plot_length,i], color='green')
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.savefig('output/transformer-step%d-feat%d.png'%(step,i))
        plt.close()
    return True

# predict the next n steps based on the input data 
def predict_future(src, prediction, nodeIdx):       
    # (batch-size, seq-len , features-num)
    # input : [ m,m+1,...,m+n ] -> [m,m+1,..., m+n+output_window_size]
    obs_length = src.size(1)
    output = torch.cat((src[nodeIdx], prediction[nodeIdx]), dim=0).cpu()

    for i in range(output.size()[1]):
        # I used this plot to visualize if the model pics up any long therm structure within the data.
        plt.plot(output[:,i],color="red")       
        plt.plot(output[:obs_length,i],color="blue")    
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.savefig('output/transformer-future-feat%d.png'%i)
        # plt.show()
        plt.close()
    return True



def expectile_loss(pred, target, expectile_level):
    """
    taking the maximum of two terms:(expectile_level - 1) * abs_errors and expectile_level * errors.
    """
    errors = target - pred
    abs_errors = torch.abs(errors)
    expectile_loss = torch.mean(torch.max((expectile_level - 1) * abs_errors, expectile_level * errors))
    return expectile_loss

# evaluating eval accuracy
def masked_huber_loss(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss_square = 0.5 * loss * loss
    loss = torch.where(loss < 1, loss_square, loss)
    return torch.mean(loss)

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


# evaluating forecasing accuracy
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / ((y_true + y_pred) / 2))) * 100

def evaluate_forecast(y_true, y_pred):
    """
    Mean Absolute Error (MAE): This metric computes the average of the absolute differences between the predicted and actual values.
    It gives an idea of the magnitude of the errors without considering the direction (positive or negative) of the errors.
    Lower values indicate better accuracy.

    Mean Squared Error (MSE): This metric calculates the average of the squared differences between the predicted and actual values.
    By squaring the errors, it emphasizes larger errors more than smaller ones, making it sensitive to outliers.
    Lower values indicate better accuracy.

    Root Mean Squared Error (RMSE): This metric is the square root of the Mean Squared Error (MSE).
    RMSE has the same unit as the original values, which makes it easier to interpret than MSE.
    Lower values indicate better accuracy.

    Mean Absolute Percentage Error (MAPE): This metric computes the average of the absolute percentage differences between the predicted and actual values.
    It provides an error measurement in percentage terms, which can be easier to understand and compare across different scales.
    Lower values indicate better accuracy. However, it has limitations when dealing with zero or near-zero values in the actual data.

    Symmetric Mean Absolute Percentage Error (sMAPE): This metric is a modified version of MAPE, which addresses some of its limitations.
    It calculates the average of the absolute percentage differences between the predicted and actual values, relative to the average of the predicted and actual values.
    This makes it symmetric and more robust when dealing with zero or near-zero values. Lower values indicate better accuracy.
    """
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    # print("Mean Absolute Percentage Error (MAPE):", mape)
    print("Symmetric Mean Absolute Percentage Error (sMAPE):", smape)


# functions for MCAGE
def consistency_loss(feature_list, metric):
    if metric == 'MSE':
        distance_metric = nn.MSELoss()
    elif metric == 'KL':
        distance_metric = nn.KLDivLoss(reduction='batchmean')
        log_softmax = nn.LogSoftmax(dim=1)

    num_features = len(feature_list)
    total_loss = 0

    for i in range(num_features):
        for j in range(i + 1, num_features):
            if metric == 'MSE':
                total_loss += distance_metric(feature_list[i], feature_list[j])
            elif metric == 'KL':
                log_prob_i = log_softmax(feature_list[i])
                prob_j = torch.exp(log_softmax(feature_list[j]))  # Convert log-softmax back to probabilities
                total_loss += distance_metric(log_prob_i, prob_j)

    avg_loss = total_loss / (num_features * (num_features - 1) / 2)
    return avg_loss

 
def get_M(adj, t=2):
    adj_numpy = adj.cpu().numpy()
    # t_order
    # t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)
    

def encode_onehot(labels):
    classes = set(labels.T)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels.T)),
                             dtype=np.int32)
    return labels_onehot

def multi_label(labels):
    def myfunction(x):
        return list(map(int, x))
    return np.apply_along_axis(myfunction, axis=0, arr=labels)


# eval funcs for MCAGE
def rank_predict(data, x, ranks):
    # query_idx is the idx of positive score
    query_idx = x.shape[0] - 1
    # sort all scores with descending, because more plausible triple has higher score
    _, idx = torch.sort(x, descending=True)
    rank = list(idx.cpu().numpy()).index(query_idx) + 1
    ranks.append(rank)
    # update data
    if rank <= 10:
        data['Hits@10'] += 1
    if rank <= 5:
        data['Hits@5'] += 1
    if rank == 1:
        data['Hits@1'] += 1
    data['MRR'] += 1.0 / rank

def test(pos_edge_index, neg_edge_index, z):
    pos_score = torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1))
    # neg_edge_index = negative_sampling(test_edges, num_nodes=len(z),num_neg_samples=5,force_undirected=False)
    neg_score = torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1))

    y_true = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
    y_score = torch.cat([pos_score, neg_score]).cpu().numpy()
    y_pred_binarized = [1 if y > 0.5 else 0 for y in y_score]

    # initial return data of validation
    data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
    ranks = []
    x = torch.cat([pos_score, neg_score], 0).squeeze()
    rank_predict(data, x, ranks)
    print("MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
        data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))


    precision = precision_score(y_true, y_pred_binarized)
    recall = recall_score(y_true, y_pred_binarized)
    f1 = f1_score(y_true, y_pred_binarized)
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    return precision, recall, f1, roc_auc, pr_auc

def predict_link(z, node_a, node_b):
    score = torch.sigmoid((z[node_a] * z[node_b]).sum())

    return score.item()

def get_top_n_predictions_wThreshold(pos_edge_index, z, threshold=0.5, k=5):
    # use the dot product as the similarity metric
    similarity_matrix = torch.sigmoid(torch.matmul(z, z.t())) # torch.matmul(z, z.t())

    # Create adjacency matrix from edge_index
    adjacency_matrix = torch.zeros_like(similarity_matrix)
    adjacency_matrix[pos_edge_index[0], pos_edge_index[1]] = 1
    adjacency_matrix[pos_edge_index[1], pos_edge_index[0]] = 1

    # Remove existing edges from the similarity matrix
    similarity_matrix_filtered = similarity_matrix * (1 - adjacency_matrix)

    # Apply threshold to the filtered similarity matrix
    similarity_matrix_filtered[similarity_matrix_filtered < threshold] = 0

    # Get the top-n predictions
    values, indices = torch.topk(similarity_matrix_filtered.view(-1), k)
    # Filter out the results with scores lower than the threshold
    mask = values >= threshold
    filtered_indices = indices[mask]
    filtered_values = values[mask]
    row_indices, col_indices = filtered_indices // similarity_matrix_filtered.size(0), filtered_indices % similarity_matrix_filtered.size(0)

    return torch.stack((row_indices, col_indices), dim=1), filtered_values

def boxplot(data):
    # Create a boxplot
    plt.boxplot(data)
    # # Save the figure
    # plt.savefig('boxplot.png')
    # Optionally, you can specify the DPI and the format
    plt.savefig('output/boxplot_highres.png', dpi=300, format='png')

def training_plot(epochs, train_loss):
    # Plotting training loss
    plt.figure() #figsize=(12, 6)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # # Plotting training accuracy
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, train_performance, 'r', label='Training accuracy')
    # plt.title('Training accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()

    plt.tight_layout()
    plt.savefig('output/trainloss_highres.png', dpi=300, format='png')


# util func
def scale_wozeros(arr, scaler):
    # Replace zeros with NaN
    arr = np.where(arr==0, np.nan, arr)
    # Perform fit_transform on non-NaN values
    scaled_arr = scaler.fit_transform(arr)
    # Replace NaNs back with zeros
    scaled_arr = np.where(np.isnan(scaled_arr), 0, scaled_arr)
    return scaled_arr


# developing code
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

def mask_test_edges(adj,edges):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    # adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    # adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj).sum() == 0

    # adj_triu = sp.triu(adj)
    # adj_tuple = sparse_to_tuple(adj_triu)
    # edges = adj_tuple[0]
    # edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    #assert ~ismember(test_edges_false, edges_all)
    #assert ~ismember(val_edges_false, edges_all)
    #assert ~ismember(val_edges, train_edges)
    #assert ~ismember(test_edges, train_edges)
    #assert ~ismember(val_edges, test_edges)

    # data = np.ones(train_edges.shape[0])

    # # Re-build adj matrix
    # adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    # adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def store_feat_csv(data):
    import csv
    with open('vector.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(['nodeID', 'featureArr'])  # Adjust the column names as needed

        # Write the vector data
        for node_id, features in enumerate(data.detach().cpu().numpy().tolist()):
            feature_array = ','.join(str(feature) for feature in features)
            writer.writerow([node_id, feature_array])


def memory_usage_in_MB(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4 / (1024 ** 2)  # Convert bytes to megabytes
