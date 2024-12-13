import os
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, TransformerConv, GATConv, Node2Vec
from torch_geometric.loader import NeighborLoader, DataLoader
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import logging
import psutil
import pandas as pd
import time
import gc
import argparse
import random
from tqdm import tqdm  # For progress bars

# -----------------------------
# Section 0: Setup Logging
# -----------------------------
logging.basicConfig(
    filename='benchmark_log.txt',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# models

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
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2):
        super(GraphSAGEModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_feats, hidden_feats))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_feats, hidden_feats))
        self.convs.append(SAGEConv(hidden_feats, out_feats))

    def forward(self, x,edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x

class GraphGPSModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2, num_heads=4):
        super(GraphGPSModel, self).__init__()
        if hidden_feats % num_heads != 0:
            raise ValueError("hidden_feats must be divisible by num_heads for GraphGPS.")
        self.linear_in = nn.Linear(in_feats, hidden_feats)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.layers.append(GCNConv(hidden_feats, hidden_feats))
            else:
                self.layers.append(TransformerConv(hidden_feats, hidden_feats, heads=num_heads, concat=False))
        self.linear_out = nn.Linear(hidden_feats, out_feats)

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
        if hidden_feats % num_heads != 0:
            raise ValueError("hidden_feats must be divisible by num_heads for GAT.")
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_feats, hidden_feats // num_heads, heads=num_heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_feats, hidden_feats // num_heads, heads=num_heads))
        
        self.convs.append(GATConv(hidden_feats, out_feats, heads=1, concat=False))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x

def load_model(model_name, in_feats, out_feats, dataset_name, num_layers=3):
    if dataset_name == 'ogbn-arxiv':
        hidden_dim = 64
        num_heads = 4
    elif dataset_name == 'ogbn-products':
        hidden_dim = 32
        num_heads = 2
    else:
        hidden_dim = 64
        num_heads = 4

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


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")
    print(f"Random seed set to {seed}")

@torch.no_grad()
def evaluate_multiclass(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    f1_scores = []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index )
        preds = out.argmax(dim=-1).cpu()
        targets = batch.y.squeeze(-1).cpu()
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        f1_scores.append(f1_score(targets, preds, average='macro'))
    acc = correct / total
    f1 = np.mean(f1_scores)
    return acc, f1

def train_multiclass(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        y_true = batch.y.squeeze(-1).long()
        out = model(batch.x, batch.edge_index)
        loss = loss_fn(out, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if step % 50 == 0:
            logger.info(f"  [Train Step {step}] Loss: {loss.item():.4f}")
            print(f"  [Train Step {step}] Loss: {loss.item():.4f}")
    return total_loss / len(loader)


def run_experiment(model_name, dataset_name, device, epochs=10, batch_size=1024, num_neighbors=[15, 10], hidden_dim=32, num_layers=3, num_heads=2):
    logger.info(f"Starting Experiment: Model={model_name}, Dataset={dataset_name}")
    print(f"\n=== Experiment: Model={model_name}, Dataset={dataset_name} ===")
    
    # Load dataset
    try:
        dataset = PygNodePropPredDataset(name=dataset_name, root='dataset/')
        data = dataset[0]
        logger.info(f"Dataset {dataset_name} loaded successfully.")
        print(f"Dataset {dataset_name} loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        print(f"Failed to load dataset {dataset_name}: {e}")
        return []
    
    # Get splits
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].squeeze()
    valid_idx = split_idx['valid'].squeeze()
    test_idx = split_idx['test'].squeeze()
    data = data.to('cpu')
    
    # going to use node2vec
    if not hasattr(data, 'x') or data.x is None:
        logger.info("No node features found. Generating Node2Vec embeddings.")
        print("No node features found. Generating Node2Vec embeddings.")
        data.x = generate_node2vec_features(data.edge_index, data.num_nodes, embedding_dim=hidden_dim)
    else:
        logger.info(f"Node features found with shape {data.x.shape}.")
        print(f"Node features found with shape {data.x.shape}.")
    
    evaluator = Evaluator(name=dataset_name)
    
    try:
        if dataset_name in ['ogbn-arxiv', 'ogbn-products']:
            out_feats = int(data.y.max()) + 1
        else:
            raise ValueError(f"Unsupported dataset {dataset_name}")
        
        model = load_model(model_name, in_feats=data.x.size(1), out_feats=out_feats, dataset_name=dataset_name, num_layers=num_layers)
        model = model.to(device)
        logger.info(f"Model {model_name} initialized and moved to device.")
        print(f"Model {model_name} initialized and moved to device.")
    except Exception as e:
        logger.error(f"Failed to initialize model {model_name}: {e}")
        print(f"Failed to initialize model {model_name}: {e}")
        return []
    # need to use neighborloader because huge data
    train_loader = NeighborLoader(
        data,
        input_nodes=train_idx,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True
    )
    
    valid_loader = DataLoader(
        torch.arange(valid_idx.size(0)),
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        torch.arange(test_idx.size(0)),
        batch_size=batch_size,
        shuffle=False
    )
    
    logger.info("DataLoaders for train, validation, and test sets initialized.")
    print("DataLoaders for train, validation, and test sets initialized.")
    
    epoch_results = []
    for epoch in range(1, epochs + 1):
        start_epoch = time.time()
        loss = train_multiclass(model, train_loader, optimizer, device)
        end_epoch = time.time()
        epoch_time = end_epoch - start_epoch
        logger.info(f"Epoch {epoch}: Loss={loss:.4f}, Time={epoch_time:.2f}s")
        print(f"Epoch {epoch}: Loss={loss:.4f}, Time={epoch_time:.2f}s")
        
        # validation
        val_acc, val_f1 = evaluate_multiclass(model, valid_loader, device)
        logger.info(f"Epoch {epoch}: Validation Accuracy={val_acc:.4f}, Validation F1 (Macro)={val_f1:.4f}")
        print(f"Epoch {epoch}: Validation Accuracy={val_acc:.4f}, Validation F1 (Macro)={val_f1:.4f}")
        
        # test
        test_acc, test_f1 = evaluate_multiclass(model, test_loader, device)
        logger.info(f"Epoch {epoch}: Test Accuracy={test_acc:.4f}, Test F1 (Macro)={test_f1:.4f}")
        print(f"Epoch {epoch}: Test Accuracy={test_acc:.4f}, Test F1 (Macro)={test_f1:.4f}")
        
        epoch_results.append({
            'dataset': dataset_name,
            'model': model_name,
            'epoch': epoch,
            'val_acc': val_acc,
            'val_f1_macro': val_f1,
            'test_acc': test_acc,
            'test_f1_macro': test_f1
        })

    
    
    logger.info(f"Completed Experiment: Model={model_name}, Dataset={dataset_name}")
    print(f"=== Completed Experiment: Model={model_name}, Dataset={dataset_name} ===")
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_results

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
        print(f"Node2Vec Epoch {epoch+1}, Loss: {total_loss:.4f}")

    node2vec.eval()
    embeddings = node2vec(torch.arange(num_nodes, device=device)).detach().cpu()
    return embeddings

def main():
    parser = argparse.ArgumentParser(description="Benchmark GNN Models on OGB Datasets")
    parser.add_argument('--models', nargs='+', default=['GCN', 'GraphSAGE', 'GraphGPS', 'GAT'],
                        help="List of models to benchmark")
    parser.add_argument('--datasets', nargs='+', default=['ogbn-arxiv', 'ogbn-products'],
                        help="List of OGB datasets to use")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=1024,
                        help="Batch size for training")
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help="Hidden dimension size")
    parser.add_argument('--num_layers', type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument('--num_heads', type=int, default=2,
                        help="Number of attention heads for GraphGPS and GAT")
    parser.add_argument('--num_neighbors', nargs='+', type=int, default=[15, 10],
                        help="Number of neighbors to sample at each layer for NeighborLoader")
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                        help="Output CSV file for results")
    args = parser.parse_args()

    set_seed(1)
    # make sur gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")
    
    all_epoch_results = []
    for dataset_name in args.datasets:
        for model_name in args.models:
            # Adjust epochs for 'ogbn-products'
            current_epochs = args.epochs
            if dataset_name == 'ogbn-products':
                current_epochs = 2  # Reduced epochs for 'ogbn-products'
            
            epoch_results = run_experiment(
                model_name=model_name,
                dataset_name=dataset_name,
                device=device,
                epochs=current_epochs,
                batch_size=args.batch_size,
                num_neighbors=args.num_neighbors,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads
            )
            if epoch_results:
                all_epoch_results.extend(epoch_results)
    
    # save here
    if all_epoch_results:
        results_df = pd.DataFrame(all_epoch_results)
        desired_columns = ['dataset', 'model', 'epoch', 'val_acc', 'val_f1_macro', 'test_acc', 'test_f1_macro']
        results_df = results_df[desired_columns]
        if not os.path.exists(args.output):
            results_df.to_csv(args.output, index=False)
        else:
            results_df.to_csv(args.output, mode='a', index=False, header=False)
        print(f"\nBenchmarking completed! Results saved to {args.output}")
        logger.info(f"Benchmarking completed! Results saved to {args.output}")
    else:
        print("\nNo experiments were completed successfully.")
        logger.warning("No experiments were completed successfully.")

if __name__ == "__main__":
    main()
