import os
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import Node2Vec,GCNConv,SAGEConv,TransformerConv,GATConv
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import numpy as np
import logging
import psutil
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Models

class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2):
        super(GCNModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_feats, hidden_feats))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_feats, hidden_feats))
        self.convs.append(GCNConv(hidden_feats, out_feats))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x

class GraphSAGEModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2 ):
        super(GraphSAGEModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_feats, hidden_feats))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_feats, hidden_feats))
        self.convs.append(SAGEConv(hidden_feats, out_feats))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x

class GraphGPSModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2, num_heads=4):
        super(GraphGPSModel, self).__init__()
        self.linear_in = nn.Linear(in_feats, hidden_feats)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.layers.append(TransformerConv(hidden_feats, hidden_feats, heads=num_heads, concat=False))
            else:
                self.layers.append(GCNConv(hidden_feats, hidden_feats))
        self.linear_out = nn.Linear(hidden_feats,out_feats)

    def forward(self, x, edge_index):
        x = self.linear_in(x)
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        x = self.linear_out(x)
        return x

class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2, num_heads=4):
        super(GATModel, self).__init__()
        self.linear_in = nn.Linear(in_feats, hidden_feats)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GATConv(hidden_feats, hidden_feats // num_heads, heads=num_heads, concat=False))
        self.linear_out = nn.Linear(hidden_feats, out_feats)
    def forward(self, x, edge_index):
        x = self.linear_in(x)
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.linear_out(x)
        return x

def load_model(model_name, in_feats, out_feats):
    #default params
    hidden_dim = 32
    num_heads = 2
    num_layers = 3
    
    if model_name == 'GCN':
        return GCNModel(in_feats, hidden_dim, out_feats, num_layers)
    elif model_name == 'GraphSAGE':
        return GraphSAGEModel(in_feats, hidden_dim, out_feats, num_layers)
    elif model_name == 'GraphGPS':
        return GraphGPSModel(in_feats, hidden_dim, out_feats, num_layers, num_heads=num_heads)
    elif model_name == 'GAT':
        return GATModel(in_feats, hidden_dim, out_feats, num_layers, num_heads=num_heads)
    else:
        raise ValueError(f"Unknown model {model_name}")

############


@torch.no_grad()
def evaluate_multilabel_ogbproteins(model, data, idx, evaluator, device):
    if not torch.is_tensor(idx):
        idx = torch.tensor(idx, dtype=torch.long)
    model_cpu = model.to('cpu')
    model_cpu.eval()
    x_cpu = data.x.cpu()
    edge_index_cpu = data.edge_index.cpu()
    out = model_cpu(x_cpu, edge_index_cpu)  # [num_nodes, 112]
    y_true = data.y[idx].cpu().float()
    y_pred = out[idx].cpu().float()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result = evaluator.eval(input_dict)
    model.to(device)
    return result['rocauc']

def train_multilabel_ogbproteins(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        y_true = batch.y.float()
        loss = loss_fn(out, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if step % 50 == 0:
            logger.info(f"  [Train Step {step}] Loss: {loss.item():.4f}")
    return total_loss / len(loader)

# Node2Vec for embeddings for the proteins dataset, since it doesn't seem like they have embeddings by default

def generate_node2vec_features(edge_index, num_nodes, embedding_dim=64, walk_length=20, context_size=10, walks_per_node=10, num_epochs=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node2vec = Node2Vec(edge_index, embedding_dim=embedding_dim, walk_length=walk_length,
                        context_size=context_size, walks_per_node=walks_per_node,
                        num_nodes=num_nodes, sparse=True).to(device)
    loader = node2vec.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

    node2vec.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Node2Vec Epoch {epoch+1}, Loss: {total_loss:.4f}")

    node2vec.eval()
    embeddings = node2vec(torch.arange(num_nodes, device=device)).detach().cpu()  # detach to avoid graph references
    return embeddings


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running on device: {device}")

    d_name = 'ogbn-proteins'
    models = ['GCN', 'GraphSAGE', 'GraphGPS', 'GAT']  # Replaced 'DRew' with 'GAT'
    current_epochs = 1

    results_csv = "benchmark_results.csv"
    file_exists = os.path.exists(results_csv)
    results = []

    logger.info(f"=== Loading dataset: {d_name} ===")
    dataset = PygNodePropPredDataset(name=d_name)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    data = dataset[0]

    edge_index = data.edge_index
    num_nodes = data.num_nodes
    logger.info("Generating Node2Vec embeddings...")
    embeddings = generate_node2vec_features(edge_index, num_nodes, embedding_dim=64, num_epochs=1)
    data.x = embeddings  # embeddings are detached, no gradient

    evaluator = Evaluator(name='ogbn-proteins')

    logger.info(f"Number of nodes: {data.x.size(0)}")
    logger.info(f"Number of features: {data.x.size(1)}")
    logger.info(f"Number of edges: {data.edge_index.size(1)}")
    logger.info(f"Number of tasks (labels): {data.y.size(1)}") 
    logger.info(f"Train/Valid/Test split sizes: {len(train_idx)}, {len(valid_idx)}, {len(test_idx)}")

    neighbor_sizes =  [10, 5, 2]
    batch_size = 512

    logger.info("Creating NeighborLoader for training...")
    train_loader = NeighborLoader(
        data,
        input_nodes=train_idx,
        num_neighbors=neighbor_sizes,
        batch_size=batch_size,
        shuffle=True
    )

    data = data.to('cpu')
    out_dim = data.y.size(1)  # 112 classes

    for model_name in models:
        logger.info(f"=== Training Model: {model_name} on {d_name} ===")
        model = load_model(model_name, data.x.size(-1), out_dim)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_val_metric = -1
        for epoch in range(1, current_epochs+1):
            logger.info(f"[Epoch {epoch}]")
            loss = train_multilabel_ogbproteins(model, train_loader, optimizer, device)
            val_rocauc = evaluate_multilabel_ogbproteins(model, data, valid_idx, evaluator, device)
            logger.info(f"  Epoch {epoch}: Loss={loss:.4f}, Val_ROC-AUC={val_rocauc:.4f}")
            if val_rocauc > best_val_metric:
                best_val_metric = val_rocauc
                test_rocauc = evaluate_multilabel_ogbproteins(model, data, test_idx, evaluator, device)
                results.append({
                    'dataset': d_name,
                    'model': model_name,
                    'epoch': epoch,
                    'val_rocauc': val_rocauc,
                    'test_rocauc': test_rocauc
                })


        logger.info(f"=== Finished Training Model: {model_name} on {d_name} ===")

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv, mode='a', index=False, header=not file_exists)

    logger.info("All experiments completed.")
