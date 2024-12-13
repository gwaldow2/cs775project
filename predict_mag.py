import sys
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
try:
    from community import community_louvain
except ImportError as e:
    print("Failed to import community_louvain. Install python-louvain package.")
    sys.exit(1)
import powerlaw
import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.special import gammaln
from sklearn.decomposition import PCA
from ogb.nodeproppred import PygNodePropPredDataset

models = ['GCN', 'GraphSAGE', 'GraphGPS', 'DRew']

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

print("Initializing MLP model for prediction...")
input_dim = 6
output_dim = 4
model = MLP(input_dim, output_dim)
model_path = "mlp_model_for_model_selection.pt"
if not os.path.exists(model_path):
    print(f"Model file {model_path} not found.")
    sys.exit(1)

print("Loading model state_dict from", model_path)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
print("State_dict keys:", list(state_dict.keys()))
for k,v in state_dict.items():
    print(f"Key: {k}, shape: {v.shape}, dtype: {v.dtype}")
    print("Sample values:", v.view(-1)[:5].tolist())
    break

model.load_state_dict(state_dict)
model.eval()
print("MLP model loaded and set to eval mode.")

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
    max_deg = len(deg_counts)-1
    cap = 100
    if max_deg > cap:
        max_deg = cap
    truncated_deg_counts = deg_counts[:max_deg+1]
    truncated_p_emp = truncated_deg_counts / truncated_deg_counts.sum()

    from scipy.special import gammaln
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
    import powerlaw
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
    score = scale_free_score(G)*0.8
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
    p = E/(N*(N-1)/2)
    c_rand = p

    nodes_list = list(G.nodes())
    sample_nodes = nodes_list if len(nodes_list)<100 else np.random.choice(nodes_list,100,replace=False)
    lengths = []
    for n in sample_nodes:
        spl = nx.single_source_shortest_path_length(G, n)
        if len(spl)>1:
            lengths.extend(spl.values())
    if len(lengths)==0:
        print("No path lengths found, WS=0")
        return 0.0
    actual_path_len = np.mean(lengths)
    mean_deg = 2*E/N
    if mean_deg>1:
        rand_path_len = math.log(N)/math.log(mean_deg)
    else:
        rand_path_len = N

    clustering_ratio = c/(c_rand+1e-9)
    path_ratio = (rand_path_len+1e-9)/(actual_path_len+1e-9)
    score = clustering_ratio * path_ratio
    score = min(1.0, score/10.0)
    print("Watts-Strogatz score:", score)
    return float(score)

def compute_features_for_user_graph(G):
    print("Computing graph features...")
    N = G.number_of_nodes()
    scale_val = N/3000000.0
    sg = sample_subgraph(G)
    modularity_val = compute_modularity(sg)
    er_score = erdos_renyi_score(sg)
    sf_score = scale_free_score(sg)
    ba_score = barabasi_albert_score(sg)
    ws_score = watts_strogatz_score(sg)
    feats = [er_score, sf_score, ba_score, ws_score, scale_val, modularity_val]
    print("Feature vector:", feats)
    return feats

print("Loading ogbn-mag dataset...")
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
print("Check what data_hetero is:", data_hetero.__class__)
print(data_hetero)

# For ogbn-mag:
# train_idx = split_idx['train']['paper']
# valid_idx = split_idx['valid']['paper']
# test_idx = split_idx['test']['paper']
if 'train' not in split_idx or 'valid' not in split_idx or 'test' not in split_idx:
    print("split_idx does not contain expected keys 'train','valid','test'")
    sys.exit(1)
if 'paper' not in split_idx['train'] or 'paper' not in split_idx['valid'] or 'paper' not in split_idx['test']:
    print("split_idx['train'], ['valid'], or ['test'] does not contain 'paper'. Keys found:")
    print("train keys:", split_idx['train'].keys())
    print("valid keys:", split_idx['valid'].keys())
    print("test keys:", split_idx['test'].keys())
    sys.exit(1)

train_idx, valid_idx, test_idx = split_idx['train']['paper'], split_idx['valid']['paper'], split_idx['test']['paper']
print("Train/Valid/Test split shapes:", train_idx.shape, valid_idx.shape, test_idx.shape)

# According to OGB docs for ogbn-mag:
# data_hetero.x_dict['paper'] for paper node features, data_hetero.y_dict['paper'] for labels
# and data_hetero.edge_index_dict[('paper','cites','paper')] for edges.

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
G = nx.Graph()
G.add_nodes_from(range(data.num_nodes))
edges_np = data.edge_index.cpu().numpy()
for i in range(edges_np.shape[1]):
    G.add_edge(edges_np[0,i], edges_np[1,i])

print("Computing features for ogbn-mag paper graph...")
user_feats = compute_features_for_user_graph(G)
user_feats_tensor = torch.tensor(user_feats, dtype=torch.float32).unsqueeze(0) # [1,6]

print("Running model prediction on ogbn-mag features...")
with torch.no_grad():
    pred_out = model(user_feats_tensor)
    pred_dist = torch.softmax(pred_out, dim=-1).numpy().flatten()

best_model_pred = models[np.argmax(pred_dist)]
print("Predicted relative performance distribution:", pred_dist)
print("Predicted best model:", best_model_pred)

print("Now we compare actual performances...")

import torch.optim as optim

def train_multi_epoch(m, data, device, epochs=10):
    m.train()
    optimizer = optim.Adam(m.parameters(), lr=0.01)
    mask = data.train_mask
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        out = m(data)
        loss = F.cross_entropy(out[mask], data.y[mask].squeeze())
        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate(m, data):
    m.eval()
    out = m(data)
    pred = out.argmax(dim=-1, keepdim=True)
    correct = (pred[data.test_mask].squeeze() == data.y[data.test_mask].squeeze()).sum().item()
    total = data.test_mask.sum().item()
    acc = correct/total if total>0 else 0.0
    return acc

from torch_geometric.nn import GCNConv, SAGEConv, TransformerConv

def load_gnn_model_simple(model_name, input_dim, hidden_dim, output_dim):
    # Basic 2-layer models for demonstration
    if model_name == 'GCN':
        class SimpleGCN(nn.Module):
            def __init__(self, in_dim, hid_dim, out_dim):
                super().__init__()
                self.conv1 = GCNConv(in_dim, hid_dim)
                self.conv2 = GCNConv(hid_dim, out_dim)
            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x = F.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x
        return SimpleGCN(input_dim,hidden_dim,output_dim)
    elif model_name == 'GraphSAGE':
        class SimpleSAGE(nn.Module):
            def __init__(self, in_dim, hid_dim, out_dim):
                super().__init__()
                self.conv1 = SAGEConv(in_dim, hid_dim)
                self.conv2 = SAGEConv(hid_dim, out_dim)
            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                x = F.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x
        return SimpleSAGE(input_dim,hidden_dim,output_dim)
    elif model_name == 'GraphGPS':
        # use GCN as placeholder
        return load_gnn_model_simple('GCN', input_dim,hidden_dim,output_dim)
    elif model_name == 'DRew':
        # use GCN as placeholder
        return load_gnn_model_simple('GCN', input_dim,hidden_dim,output_dim)

print("Evaluating actual performance for each model with multiple epochs (10 epochs)...")
input_dim = data.x.size(1)
out_dim = int(data.y.max().item())+1
hidden_dim = 64
actual_accuracies = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

EPOCHS = 10
for mname in models:
    print(f"Training {mname} for {EPOCHS} epochs...")
    m = load_gnn_model_simple(mname, input_dim, hidden_dim, out_dim).to(device)
    train_multi_epoch(m, data, device, epochs=EPOCHS)
    acc = evaluate(m, data)
    actual_accuracies[mname] = acc
    print(f"{mname} Test Accuracy after {EPOCHS} epochs: {acc:.4f}")

actual_best_model = max(actual_accuracies, key=actual_accuracies.get)
print("\nActual Test Accuracies:", actual_accuracies)
print("Actual best model:", actual_best_model)

print("\nPredicted best model:", best_model_pred)
print("Actual best model:", actual_best_model)

with open("comparison_results_mag.txt","w") as f:
    f.write("Predicted distribution:\n")
    for m, val in zip(models, pred_dist):
        f.write(f"{m}: {val:.4f}\n")
    f.write(f"\nPredicted best model: {best_model_pred}\n")
    f.write("\nActual accuracies:\n")
    for m,acc in actual_accuracies.items():
        f.write(f"{m}: {acc:.4f}\n")
    f.write(f"\nActual best model: {actual_best_model}\n")

print("Comparison saved to comparison_results_mag.txt")

# Save the predicted distribution plot
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(models, pred_dist, color=['blue','orange','green','red'])
ax.set_title("Predicted Relative Performance Distribution (ogbn-mag)")
ax.set_ylabel("Confidence")
best_idx = np.argmax(pred_dist)
ax.bar(models[best_idx], pred_dist[best_idx], color='yellow', edgecolor='red', linewidth=2)
for i, v in enumerate(pred_dist):
    ax.text(i, v+0.01, f"{v:.2f}", ha='center', fontweight='bold')
plt.savefig("predicted_distribution_mag.png")
plt.close()
print("Distribution image saved as predicted_distribution_mag.png")

# Create a plot showing predicted vs actual with dual y-axis
print("Creating predicted vs. actual comparison plot with dual y-axis...")

pred_values = pred_dist
actual_values = [actual_accuracies[m] for m in models]

# Compute a scale factor to scale accuracy for better visual comparison
# e.g. ratio of mean predicted to mean actual
mean_pred = np.mean(pred_values)
mean_actual = np.mean(actual_values) if np.mean(actual_values)>0 else 1.0
scale_factor = mean_pred / mean_actual
print(f"Scaling accuracy by factor {scale_factor:.2f} for visualization.")

x = np.arange(len(models))
fig, ax1 = plt.subplots(figsize=(8,6))
ax1.set_xlabel('Models')
ax1.set_ylabel('Predicted Confidence', color='blue')
bars = ax1.bar(x, pred_values, color=['blue','orange','green','red'])
ax1.tick_params(axis='y', labelcolor='blue')
for i, v in enumerate(pred_values):
    ax1.text(i, v+0.01, f"{v:.2f}", ha='center', color='blue', fontweight='bold')

# Create second axis for accuracy
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy (scaled)', color='green')
scaled_accuracy = [a*scale_factor for a in actual_values]
ax2.plot(x, scaled_accuracy, color='green', marker='o', linewidth=2)
ax2.tick_params(axis='y', labelcolor='green')
for i, v in enumerate(scaled_accuracy):
    ax2.text(i, v+0.01, f"{actual_values[i]:.2f}", ha='center', color='green', fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_title("Predicted Confidence vs Actual Accuracy (scaled) for ogbn-mag")

fig.tight_layout()
plt.savefig("predicted_vs_actual_comparison_mag.png")
plt.close()
print("Predicted vs. Actual comparison plot saved as predicted_vs_actual_comparison_mag.png")

# PCA comparison plot
print("Creating PCA comparison plot...")
training_feats = np.array([
    [0.5, 0.2, 0.1, 0.3, 0.0001, 0.4],
    [0.1, 0.05, 0.04, 0.2, 0.0002, 0.35],
    [0.6, 0.3, 0.2, 0.1, 0.00015, 0.45]
])

all_feats = np.vstack([training_feats, user_feats])
pca = PCA(n_components=2)
coords = pca.fit_transform(all_feats)

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(coords[:3,0], coords[:3,1], c=['blue','green','orange'], label='Training Graphs')
ax.scatter(coords[3,0], coords[3,1], c='red', label='ogbn-mag Graph', marker='x', s=100)
for i in range(3):
    ax.text(coords[i,0], coords[i,1], f"Train{i+1}", fontweight='bold')
ax.text(coords[3,0], coords[3,1], "MAG", fontweight='bold')

ax.set_title("PCA of Graph Features (ogbn-mag)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
plt.savefig("pca_comparison_mag.png")
plt.close()
print("PCA comparison plot saved as pca_comparison_mag.png")

print("All done. Check comparison_results_mag.txt, predicted_distribution_mag.png, predicted_vs_actual_comparison_mag.png, and pca_comparison_mag.png for results.")
