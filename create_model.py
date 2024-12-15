import sys
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
try:
    from community import community_louvain
except ImportError:
    print("Failed to import community_louvain. Install python-louvain package using 'pip install python-louvain'.")
    sys.exit(1)
import powerlaw
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import gammaln
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm

models = ['GCN', 'GraphSAGE', 'GraphGPS', 'GAT']

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=4, hidden_dim=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x

def sample_subgraph(G, sample_size=10000, seed=42):
    print("Sampling subgraph if needed...")
    np.random.seed(seed)
    N = G.number_of_nodes()
    if N <= sample_size:
        print("No subgraph sampling needed.")
        return G
    nodes = list(G.nodes())
    nodes_to_keep = np.random.choice(nodes, sample_size, replace=False)
    sg = G.subgraph(nodes_to_keep).copy()
    print(f"Subgraph of {sg.number_of_nodes()} nodes sampled from {N}.")
    return sg

def compute_modularity(G):
    if G.number_of_edges() == 0:
        print("No edges, modularity=0")
        return 0.0
    partition = community_louvain.best_partition(G)
    modularity_val = community_louvain.modularity(partition, G)
    print("Modularity:", modularity_val)
    return modularity_val

def erdos_renyi_score(G):
    print("Computing Erdos-Renyi score...")
    degrees = [d for _, d in G.degree()]
    if len(degrees) == 0:
        print("No degrees, ER=0")
        return 0.0
    mean_deg = np.mean(degrees)
    if mean_deg == 0:
        print("mean_deg=0, ER=0")
        return 0.0
    deg_counts = np.bincount(degrees)
    p_emp = deg_counts / deg_counts.sum()
    max_deg = len(deg_counts) - 1
    cap = 100
    if max_deg > cap:
        max_deg = cap
    truncated_deg_counts = deg_counts[:max_deg + 1]
    truncated_p_emp = truncated_deg_counts / truncated_deg_counts.sum()
    p_pois = []
    for k in range(max_deg + 1):
        log_p = -mean_deg + k * math.log(mean_deg + 1e-15) - gammaln(k + 1)
        val = math.exp(log_p)
        p_pois.append(val)
    p_pois = np.array(p_pois)
    p_pois = p_pois / p_pois.sum()
    mask = (truncated_p_emp > 0)
    kl_div = np.sum(truncated_p_emp[mask] * np.log(truncated_p_emp[mask] / p_pois[mask]))
    score = 1 / (1 + kl_div)
    print("Erdos-Renyi score:", score)
    return float(score)

def scale_free_score(G):
    print("Computing scale-free score...")
    degrees = [d for _, d in G.degree() if d > 0]
    if len(degrees) < 10:
        print("Not enough degrees, SF=0")
        return 0.0
    try:
        fit = powerlaw.Fit(degrees, quiet=True)
        R, p = fit.distribution_compare('power_law', 'exponential')
        if p > 0.1 and R > 0:
            sf = min(p, 1.0)
            print("Scale-free score:", sf)
            return sf
        else:
            print("No scale-free pattern, SF=0")
            return 0.0
    except Exception as e:
        print("Failed to compute scale-free score, returning 0:", e)
        return 0.0

def barabasi_albert_score(G):
    score = scale_free_score(G) * 0.8
    print("Barabasi-Albert score:", score)
    return score

def watts_strogatz_score(G):
    print("Computing Watts-Strogatz score...")
    if G.number_of_nodes() < 10 or G.number_of_edges() == 0:
        print("Graph too small or no edges, WS=0")
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
        print("No path lengths found, WS=0")
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
    score = min(1.0, score /10.0)
    print("Watts-Strogatz score:", score)
    return float(score)

def compute_features_for_dataset(d_name):
    print(f"\nComputing features for {d_name}...")
    try:
        dataset = PygNodePropPredDataset(name=d_name, root='dataset/')
    except Exception as e:
        print(f"Error loading dataset {d_name}: {e}")
        sys.exit(1)
    data = dataset[0]
    N = data.num_nodes
    scale_val = N / 3000000.0
    sample_size = 10000 if N > 10000 else N
    print("Converting edge_index to NetworkX Graph...")
    G = nx.Graph()
    G.add_nodes_from(range(N))
    edges_np = data.edge_index.cpu().numpy()
    G.add_edges_from(zip(edges_np[0], edges_np[1]))
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    G = sample_subgraph(G, sample_size=sample_size)
    modularity_val = compute_modularity(G)
    er_score = erdos_renyi_score(G)
    sf_score = scale_free_score(G)
    ba_score = barabasi_albert_score(G)
    ws_score = watts_strogatz_score(G)
    return [er_score, sf_score, ba_score, ws_score, scale_val, modularity_val]

results_csv = "benchmark_results.csv"
if not os.path.exists(results_csv):
    print(f"CSV file {results_csv} not found.")
    sys.exit(1)

df = pd.read_csv(results_csv)
metric_f1 = 'test_f1_macro'
metric_rocauc = 'test_rocauc'
    
best_results_f1 = df.groupby(['dataset', 'model'])[metric_f1].max().reset_index()
best_results_rocauc = df.groupby(['dataset', 'model'])[metric_rocauc].max().reset_index()
multi_class_datasets = ['ogbn-arxiv', 'ogbn-products']
multi_label_datasets = ['ogbn-proteins']
unique_datasets = df['dataset'].unique()
dataset_features = {}
for dset in unique_datasets:
    feats = compute_features_for_dataset(dset)
    dataset_features[dset] = {
        'Erdos_Renyi': feats[0],
        'Scale_free': feats[1],
        'Barabasi_Albert': feats[2],
        'Watts_Strogatz': feats[3],
        'Scale': feats[4],
        'Modularity': feats[5]
    }

features_df = pd.DataFrame.from_dict(dataset_features, orient='index')
features_df.reset_index(inplace=True)
features_df.rename(columns={'index': 'dataset'},inplace=True)
features_df.to_csv("computed_features.csv",index=False)
print("\nComputed graph features saved to computed_features.csv")

X = []
Y = []

for dset in unique_datasets:
    feats_dict = dataset_features[dset]
    feats = [
        feats_dict['Erdos_Renyi'],
        feats_dict['Scale_free'],
        feats_dict['Barabasi_Albert'],
        feats_dict['Watts_Strogatz'],
        feats_dict['Scale'],
        feats_dict['Modularity']
    ]
    metrics = []
    if dset in multi_class_datasets:
        dset_data = best_results_f1[best_results_f1['dataset'] == dset]
        for model in models:
            row = dset_data[dset_data['model'] == model]
            if not row.empty and not pd.isna(row[metric_f1].values[0]):
                metrics.append(row[metric_f1].values[0])
            else:
                metrics.append(0.0)
    elif dset in multi_label_datasets:
        dset_data = best_results_rocauc[best_results_rocauc['dataset'] == dset]
        for model in models:
            row = dset_data[dset_data['model'] == model]
            if not row.empty and not pd.isna(row[metric_rocauc].values[0]):
                metrics.append(row[metric_rocauc].values[0])
            else:
                metrics.append(0.0)
    else:
        print(f"Dataset {dset} not recognized for metric selection.")
        continue
    metrics = np.array(metrics, dtype=np.float32)
    sum_metrics = metrics.sum()
    if sum_metrics > 0:
        relative_metrics = metrics / sum_metrics
    else:
        relative_metrics = np.array([1.0 / len(models)] * len(models), dtype=np.float32)
    X.append(feats)
    Y.append(relative_metrics)

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

print(f"\nTotal training samples: {X.shape[0]}")
print(f"Feature vector shape: {X.shape}")
print(f"Target vector shape: {Y.shape}")

input_dim = X.shape[1]
output_dim = len(models)

print("\nInitializing MLP model for prediction...")
model = MLP(input_dim, output_dim, hidden_dim=64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

epochs = 100
print("\nStarting MLP training...")
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")

model_save_path = "mlp_model_for_model_selection.pt"
torch.save(model.state_dict(), model_save_path)
print(f"\nMLP model saved to {model_save_path}")

model.eval()
with torch.no_grad():
    predicted = model(X_tensor).cpu().numpy()

actual = Y_tensor.cpu().numpy()
global_max = max(predicted.max(), actual.max())
print(f"\nGlobal maximum relative performance across all datasets and models: {global_max:.3f}")
