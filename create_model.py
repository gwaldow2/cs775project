import os
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
import networkx as nx
from community import community_louvain  # pip install python-louvain
import powerlaw  # pip install powerlaw
import math
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

###################################
# Load CSV Results
###################################

results_csv = "combined_benchmark_results.csv"  # Ensure this CSV is correctly formatted
df = pd.read_csv(results_csv)

# Set the scoring metric to 'test_f1_macro'
metric_col = 'test_f1_macro'

# Define the models used
models = ['GCN', 'GraphSAGE', 'GraphGPS', 'GAT']

# Get best results per dataset-model
best_results = df.groupby(['dataset', 'model'])[metric_col].max().reset_index()

###################################
# Graph Feature Computation Functions
###################################

def sample_subgraph(edge_index, num_nodes, sample_size=10000, seed=42):
    np.random.seed(seed)
    if num_nodes <= sample_size:
        nodes_to_keep = np.arange(num_nodes)
    else:
        nodes_to_keep = np.random.choice(num_nodes, sample_size, replace=False)
    nodes_to_keep_set = set(nodes_to_keep)

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    mask = np.isin(src, nodes_to_keep) & np.isin(dst, nodes_to_keep)
    src_f = src[mask]
    dst_f = dst[mask]

    G = nx.Graph()
    G.add_nodes_from(nodes_to_keep)
    edges = list(zip(src_f, dst_f))
    G.add_edges_from(edges)
    return G

def compute_modularity(G):
    if G.number_of_edges() == 0:
        return 0.0
    partition = community_louvain.best_partition(G)
    modularity_val = community_louvain.modularity(partition, G)
    return modularity_val

from scipy.special import gammaln

def erdos_renyi_score(G):
    degrees = [d for _, d in G.degree()]
    if len(degrees) == 0:
        return 0.0
    mean_deg = np.mean(degrees)
    if mean_deg == 0:
        return 0.0

    deg_counts = np.bincount(degrees)
    p_emp = deg_counts / deg_counts.sum()

    max_deg = len(deg_counts) - 1
    cap = 100
    if max_deg > cap:
        max_deg = cap

    truncated_deg_counts = deg_counts[:max_deg+1]
    truncated_p_emp = truncated_deg_counts / truncated_deg_counts.sum()

    p_pois = []
    for k in range(max_deg+1):
        log_p = -mean_deg + (k * math.log(mean_deg + 1e-15)) - gammaln(k+1)
        val = math.exp(log_p)
        p_pois.append(val)
    p_pois = np.array(p_pois)
    p_pois = p_pois / p_pois.sum()

    mask = (truncated_p_emp > 0)
    kl_div = np.sum(truncated_p_emp[mask] * np.log(truncated_p_emp[mask] / p_pois[mask]))
    score = 1 / (1 + kl_div)
    return float(score)

def scale_free_score(G):
    degrees = [d for _, d in G.degree() if d > 0]
    if len(degrees) < 10:
        return 0.0
    fit = powerlaw.Fit(degrees, quiet=True)
    R, p = fit.distribution_compare('power_law', 'exponential')
    if p > 0.1 and R > 0:
        return min(p, 1.0)
    else:
        return 0.0

def barabasi_albert_score(G):
    return scale_free_score(G) * 0.8

def watts_strogatz_score(G):
    if G.number_of_nodes() < 10 or G.number_of_edges() == 0:
        return 0.0
    c = nx.average_clustering(G)
    N = G.number_of_nodes()
    E = G.number_of_edges()
    p = E / (N * (N - 1) / 2)
    c_rand = p

    nodes_list = list(G.nodes())
    sample_nodes = nodes_list if len(nodes_list) < 100 else np.random.choice(nodes_list, 100, replace=False)
    lengths = []
    for n in sample_nodes:
        spl = nx.single_source_shortest_path_length(G, n)
        if len(spl) > 1:
            lengths.extend(spl.values())
    if len(lengths) == 0:
        return 0.0
    actual_path_len = np.mean(lengths)
    mean_deg = 2 * E / N
    if mean_deg > 1:
        rand_path_len = math.log(N) / math.log(mean_deg)
    else:
        rand_path_len = N

    clustering_ratio = c / (c_rand + 1e-9)
    path_ratio = (rand_path_len + 1e-9) / (actual_path_len + 1e-9)
    score = clustering_ratio * path_ratio
    score = min(1.0, score / 10.0)
    return float(score)

def compute_features_for_dataset(d_name):
    dataset = PygNodePropPredDataset(name=d_name)
    data = dataset[0]
    N = data.num_nodes

    scale_val = N / 3000000.0
    sample_size = 10000 if N > 10000 else N
    G = sample_subgraph(data.edge_index, N, sample_size=sample_size)

    modularity_val = compute_modularity(G)
    er_score = erdos_renyi_score(G)
    sf_score = scale_free_score(G)
    ba_score = barabasi_albert_score(G)
    ws_score = watts_strogatz_score(G)

    return [er_score, sf_score, ba_score, ws_score, scale_val, modularity_val]

unique_datasets = df['dataset'].unique() 

dataset_features = {}
for dset in unique_datasets:
    print(f"Computing features for {dset}...")
    feats = compute_features_for_dataset(dset)
    dataset_features[dset] = {
        'Erdos_Renyi': feats[0],
        'Scale_free': feats[1],
        'Barabasi_Albert': feats[2],
        'Watts_Strogatz': feats[3],
        'Scale': feats[4],
        'Modularity': feats[5]
    }

X = []
Y = []

for dset in unique_datasets:
    dset_data = best_results[best_results['dataset'] == dset]
    dset_models_scores = []
    for m in models:
        row = dset_data[dset_data['model'] == m]
        if len(row) == 0:
            dset_models_scores.append(0.0)
        else:
            score = row[metric_col].values[0]
            dset_models_scores.append(score)

    sum_scores = sum(dset_models_scores)
    if sum_scores == 0:
        relative_scores = [0.25, 0.25, 0.25, 0.25]
    else:
        relative_scores = [s / sum_scores for s in dset_models_scores]

    feats_dict = dataset_features[dset]
    feats = [
        feats_dict['Erdos_Renyi'],
        feats_dict['Scale_free'],
        feats_dict['Barabasi_Albert'],
        feats_dict['Watts_Strogatz'],
        feats_dict['Scale'],
        feats_dict['Modularity']
    ]
    X.append(feats)
    Y.append(relative_scores)

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

input_dim = X.shape[1]
output_dim = len(models)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

X_torch = torch.tensor(X)
Y_torch = torch.tensor(Y)

model = MLP(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

epochs = 100
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    pred = model(X_torch)
    pred_dist = torch.softmax(pred, dim=-1)
    loss = criterion(pred_dist, Y_torch)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")

save_path = "mlp_model_for_model_selection.pt"
torch.save(model.state_dict(), save_path)
print(f"MLP saved to {save_path}")

datasets_sorted = sorted(unique_datasets)
fig, axes = plt.subplots(nrows=len(datasets_sorted), ncols=1, figsize=(6, 4 * len(datasets_sorted)))

if len(datasets_sorted) == 1:
    axes = [axes]  # Ensure axes is iterable if only one dataset.

for ax, dset in zip(axes, datasets_sorted):
    dset_data = best_results[best_results['dataset'] == dset]
    dset_scores = []
    for m in models:
        row = dset_data[dset_data['model'] == m]
        if len(row) == 0:
            dset_scores.append(0.0)
        else:
            dset_scores.append(row[metric_col].values[0])
    ax.bar(models, dset_scores, color=['blue', 'orange', 'green', 'red'])
    ax.set_title(f"Dataset: {dset}")
    ax.set_ylabel(metric_col)
    ax.set_ylim([0, max(dset_scores) * 1.1 if max(dset_scores) > 0 else 1])

plt.tight_layout()
plt.savefig("model_performance.png")
print("Plot saved to model_performance.png")
plt.close()
