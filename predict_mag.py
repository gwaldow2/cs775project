import sys
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
from community import community_louvain
import powerlaw
import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.special import gammaln
from sklearn.decomposition import PCA
from ogb.nodeproppred import PygNodePropPredDataset

models = ['GCN', 'GraphSAGE', 'GraphGPS', 'GAT']

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=4, hidden_dim=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 4 outputs for 4 models

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)  # Ensure outputs sum to 1
        return x

def load_mlp_model(model_path, input_dim=6, output_dim=4):
    model = MLP(input_dim, output_dim, hidden_dim=64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        sys.exit(1)
    print("Loading MLP model state_dictfrom", model_path)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("MLP model loaded")
    return model, device

def compute_modularity(G, sample_size=100000):
    """
    Computes the modularity of the graph.
    If graph is larger than sample_size, sample a subset of nodes.
    """
    if G.number_of_edges() == 0:
        print("No edges, modularity=0")
        return 0.0
    if G.number_of_nodes() > sample_size:
        print(f"Graph too large for modularity computation ({G.number_of_nodes()} nodes). Sampling {sample_size} nodes.")
        sampled_nodes = np.random.choice(G.nodes(), sample_size, replace=False)
        G_sample = G.subgraph(sampled_nodes).copy()
    else:
        G_sample = G
    print("Running Community Louvain algorithm for modularity...")
    partition = community_louvain.best_partition(G_sample)
    modularity_val = community_louvain.modularity(partition, G_sample)
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
    p_emp = deg_counts /deg_counts.sum()
    max_deg = len(deg_counts)-1
    cap = 100
    if max_deg > cap:
        max_deg = cap
    truncated_deg_counts = deg_counts[:max_deg+1]
    truncated_p_emp = truncated_deg_counts / truncated_deg_counts.sum()

    p_pois = []
    for k in range(max_deg+1):
        log_p = -mean_deg + k*math.log(mean_deg+1e-15) - gammaln(k+1)
        val = math.exp(log_p)
        p_pois.append(val)
    p_pois = np.array(p_pois)
    p_pois = p_pois / p_pois.sum()

    mask = (truncated_p_emp > 0)
    kl_div = np.sum(truncated_p_emp[mask]*np.log(truncated_p_emp[mask]/p_pois[mask]))
    score = 1/(1+kl_div)
    print("Erdos-Renyi score:", score)
    return float(score)

def scale_free_score(G):
    print("Computing scale-free score...")
    degrees = [d for _, d in G.degree() if d>0]
    if len(degrees) < 10:
        print("Not enough degrees, SF=0")
        return 0.0
    try:
        fit = powerlaw.Fit(degrees, quiet=True)
        R, p = fit.distribution_compare('power_law', 'exponential')
        if p > 0.1 and R > 0:
            sf = min(p,1.0)
            print("Scale-free score:", sf)
            return sf
        else:
            print("No scale-free pattern, SF=0")
            return 0.0
    except Exception as e:
        print("Failed to compute scale-free score, returning 0:", e)
        return 0.0

def barabasi_albert_score(G):
    print("Computing Barabasi-Albert score...")
    score = scale_free_score(G)*0.8
    print("Barabasi-Albert score:", score)
    return score

def watts_strogatz_score(G, sample_size=100):
    print("Computing Watts-Strogatz score...")
    if G.number_of_nodes() < 10 or G.number_of_edges() == 0:
        print("Graph too small or no edges, WS=0")
        return 0.0
    c = nx.average_clustering(G)
    N = G.number_of_nodes()
    E = G.number_of_edges()
    p = E/(N*(N-1)/2)
    c_rand = p

    nodes_list = list(G.nodes())
    if len(nodes_list) < sample_size:
        sample_nodes = nodes_list
    else:
        sample_nodes = np.random.choice(nodes_list, sample_size, replace=False)
    lengths = []
    for n in sample_nodes:
        spl = nx.single_source_shortest_path_length(G, n)
        if len(spl) > 1:
            lengths.extend(spl.values())
    if len(lengths) == 0:
        print("No path lengths found, WS=0")
        return 0.0
    actual_path_len = np.mean(lengths)
    mean_deg = 2*E/N
    if mean_deg > 1:
        rand_path_len = math.log(N)/math.log(mean_deg)
    else:
        rand_path_len = N

    clustering_ratio = c/(c_rand+1e-9)
    path_ratio = (rand_path_len+1e-9)/(actual_path_len+1e-9)
    score = clustering_ratio * path_ratio
    score = min(1.0, score/10.0)
    print("Watts-Strogatz score:", score)
    return float(score)

def compute_graph_features(G):
    print("Computing graph features...")
    modularity_val = compute_modularity(G)
    er_score = erdos_renyi_score(G)
    sf_score = scale_free_score(G)
    ba_score = barabasi_albert_score(G)
    ws_score = watts_strogatz_score(G)
    scale_val = G.number_of_nodes() / 3000000.0
    feats = [er_score, sf_score, ba_score, ws_score, scale_val, modularity_val]
    print("Feature vector:", feats)
    return feats

def compute_features_for_mag():
    print("\nLoading ogbn-mag dataset...")
    try:
        dataset = PygNodePropPredDataset(name="ogbn-mag", root='dataset/')
    except Exception as e:
        print("Error loading ogbn-mag dataset:", e)
        sys.exit(1)
    
    split_idx = dataset.get_idx_split()
    print("split_idx keys:", split_idx.keys())
    for k in split_idx.keys():
        print(f"split_idx[{k}] type: {type(split_idx[k])}")
        if isinstance(split_idx[k], dict):
            print(f"split_idx[{k}] keys: {split_idx[k].keys()}")
        else:
            print(f"split_idx[{k}] is not a dict")
    
    data_hetero = dataset[0]
    print("Data hetero structure:", data_hetero)
    
    # Verify required splits and keys
    required_splits = ['train', 'valid', 'test']
    for split in required_splits:
        if split not in split_idx:
            print(f"split_idx does not contain expected key '{split}'")
            sys.exit(1)
        if 'paper' not in split_idx[split]:
            print(f"split_idx['{split}'] does not contain 'paper' key. Keys found: {split_idx[split].keys()}")
            sys.exit(1)
    
    train_idx, valid_idx, test_idx = split_idx['train']['paper'], split_idx['valid']['paper'], split_idx['test']['paper']
    print("Train/Valid/Test split shapes:", train_idx.shape, valid_idx.shape, test_idx.shape)
    
    # Extract node features and edges for 'paper'
    if not hasattr(data_hetero, 'x_dict'):
        print("data_hetero has no x_dict attribute. Dumping data_hetero:")
        print(data_hetero)
        sys.exit(1)
    
    if 'paper' not in data_hetero.x_dict:
        print("'paper' not found in data_hetero.x_dict keys:", data_hetero.x_dict.keys())
        sys.exit(1)
    
    if 'paper' not in data_hetero.y_dict:
        print("'paper' not found in data_hetero.y_dict keys:", data_hetero.y_dict.keys())
        sys.exit(1)
    
    if ('paper','cites','paper') not in data_hetero.edge_index_dict:
        print("('paper','cites','paper') not found in edge_index_dict keys:", data_hetero.edge_index_dict.keys())
        sys.exit(1)
    
    paper_x = data_hetero.x_dict['paper']
    paper_y = data_hetero.y_dict['paper']
    paper_edge_index = data_hetero.edge_index_dict[('paper','cites','paper')]
    
    print("paper_x shape:", paper_x.shape)
    print("paper_y shape:", paper_y.shape)
    print("paper_edge_index shape:", paper_edge_index.shape)
    
    from torch_geometric.data import Data
    data = Data(x=paper_x, y=paper_y, edge_index=paper_edge_index)
    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    
    valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
    valid_mask[valid_idx] = True
    
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True
    
    data.train_mask = train_mask
    data.valid_mask = valid_mask
    data.test_mask = test_mask
    
    print(f"Paper subgraph: {data.num_nodes} nodes, {data.num_edges} edges")
    
    print("Converting to networkx graph for feature computation...")
    G_mag = nx.Graph()
    G_mag.add_nodes_from(range(data.num_nodes))
    edges_np_mag = data.edge_index.cpu().numpy()
    print("Adding edges to NetworkX graph...")
    G_mag.add_edges_from(zip(edges_np_mag[0], edges_np_mag[1]))
    print("Edges added.")
    
    print("Computing features for ogbn-mag paper graph...")
    user_feats = compute_graph_features(G_mag)
    user_feats = np.array(user_feats).reshape(1, -1)  # [1,6]
    
    return user_feats

def main():
    # Load precomputed features
    computed_features_csv = "computed_features.csv"
    if not os.path.exists(computed_features_csv):
        print(f"Computed features CSV {computed_features_csv} not found. Ensure that `create_model.py` has been run.")
        sys.exit(1)
    features_df = pd.read_csv(computed_features_csv)
    print("\nLoaded computed_features.csv:")
    print(features_df.head())
    
    if 'dataset' not in features_df.columns:
        print("'dataset' column not found in computed_features.csv. Please add it with dataset names.")
        sys.exit(1)
    
    model_path = "mlp_model_for_model_selection.pt"
    model, device = load_mlp_model(model_path)
    
    user_feats = compute_features_for_mag()
    
    user_feats_tensor = torch.tensor(user_feats, dtype=torch.float32).to(device)
    print("\nRunning model prediction on ogbn-mag features...")
    with torch.no_grad():
        pred_out = model(user_feats_tensor)
        pred_dist = pred_out.cpu().numpy().flatten()
    
    best_model_pred = models[np.argmax(pred_dist)]
    print("Predicted relative performance distribution:", pred_dist)
    print("Predicted best model:", best_model_pred)
    
    # PCA
    print("\nPerforming PCA on precomputed features and ogbn-mag...")
    training_feats = features_df[['Erdos_Renyi', 'Scale_free', 'Barabasi_Albert', 'Watts_Strogatz', 'Scale', 'Modularity']].values
    dataset_names = features_df['dataset'].values
    all_feats = np.vstack([training_feats, user_feats])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_feats)
    
    # Plot PCA
    plt.figure(figsize=(10,8))
    plt.scatter(coords[:-1,0], coords[:-1,1], c='blue', label='Training Graphs', alpha=0.6)
    plt.scatter(coords[-1,0], coords[-1,1], c='red', label='ogbn-mag Graph', marker='X', s=200)
    for i, name in enumerate(dataset_names):
        plt.text(coords[i,0], coords[i,1], f"{name}", fontsize=9, ha='right')
    plt.text(coords[-1,0], coords[-1,1], "MAG", fontsize=12, ha='left', fontweight='bold')
    
    plt.title("PCA of Graph Features (Including ogbn-mag)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pca_comparison_mag.png")
    plt.close()
    print("PCA comparison plot saved as pca_comparison_mag.png")
    
    # predicted Distribution
    plt.figure(figsize=(10,6))
    bars = plt.bar(models, pred_dist, color=['blue','orange','green','red'])
    plt.xlabel('Models')
    plt.ylabel('Relative Performance')
    plt.title('Predicted Relative Performance Distribution: ogbn-mag')
    plt.ylim([0, max(pred_dist) + 0.05])  # Set y-limit based on max predicted
    
    for bar, value in zip(bars, pred_dist):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001, f"{value:.3f}", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("predicted_distribution_mag.png")
    plt.close()
    print("Predicted distribution plot saved as predicted_distribution_mag.png")
    
    print("\nAll tasks completed successfully. Check the saved `pca_comparison_mag.png` and `predicted_distribution_mag.png` for results.")

if __name__ == "__main__":
    main()
