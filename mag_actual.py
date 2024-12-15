import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.metrics import accuracy_score, f1_score
import random
import numpy as np
import os
import gc 

# Section 1: Models
# ----------------------------

class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2, dropout=0.5):
        super(GCNModel, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_feats, hidden_feats))
        self.bns.append(nn.BatchNorm1d(hidden_feats))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_feats, hidden_feats))
            self.bns.append(nn.BatchNorm1d(hidden_feats))
        self.convs.append(GCNConv(hidden_feats, out_feats))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class GraphSAGEModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2, dropout=0.5):
        super(GraphSAGEModel, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(SAGEConv(in_feats, hidden_feats))
        self.bns.append(nn.BatchNorm1d(hidden_feats))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_feats, hidden_feats))
            self.bns.append(nn.BatchNorm1d(hidden_feats))
        self.convs.append(SAGEConv(hidden_feats, out_feats))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class GraphGPSModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2, num_heads=2, dropout=0.5):
        super(GraphGPSModel, self).__init__()
        if hidden_feats % num_heads != 0:
            raise ValueError("hidden_feats must be divisible by num_heads for GraphGPS.")
        self.linear_in = nn.Linear(in_feats, hidden_feats)
        self.layers = nn.ModuleList()
        
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.layers.append(TransformerConv(hidden_feats, hidden_feats, heads=num_heads, concat=False))
            else:
                self.layers.append(GCNConv(hidden_feats, hidden_feats))
            self.bns.append(nn.BatchNorm1d(hidden_feats))
        self.linear_out = nn.Linear(hidden_feats, out_feats)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.linear_in(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_out(x)
        return x

class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout=0.3):
        super(GATModel, self).__init__()
        self.conv = GATConv(in_feats, hidden_feats, heads=1, concat=False)
        self.bn = nn.BatchNorm1d(hidden_feats)
        self.fc = nn.Linear(hidden_feats, out_feats)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x


# Section 2: Util

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_and_evaluate(model, graph, train_idx, val_idx, test_idx, device, num_classes, epochs=10, patience=3, learning_rate=0.01, weight_decay=5e-4, dropout=0.5):
    """
    Trains the model and evaluates it on validation and test sets.
    Returns the best metrics.
    """
    model = model.to(device)
    print(f"Memory Summary after moving model to device:\n{torch.cuda.memory_summary(device=device, abbreviated=True)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0

    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x,graph.edge_index)
        loss = criterion(out[train_idx], graph.y[train_idx])
        loss.backward()
        optimizer.step()

        # Valid
        model.eval()
        with torch.no_grad():
            out = model(graph.x, graph.edge_index)
            val_logits = out[val_idx]
            val_labels = graph.y[val_idx]
            val_loss = criterion(val_logits, val_labels).item()
            val_pred = val_logits.argmax(dim=-1)
            val_acc = accuracy_score(val_labels.cpu(), val_pred.cpu())
            val_f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='macro')

        print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        print(f"Memory Summary after Epoch {epoch}:\n{torch.cuda.memory_summary(device=device, abbreviated=True)}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test 
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index)
        test_logits = out[test_idx]
        test_labels = graph.y[test_idx]
        test_pred = test_logits.argmax(dim=-1)
        test_acc = accuracy_score(test_labels.cpu(), test_pred.cpu())
        test_f1 = f1_score(test_labels.cpu(), test_pred.cpu(), average='macro')

    return {
        'val_loss': best_val_loss,
        'val_acc': best_val_acc,
        'val_f1': best_val_f1,
        'test_acc': test_acc,
        'test_f1': test_f1
    }


def train_model(model_name, model_class, graph, train_idx, val_idx, test_idx, device, num_classes):
    print(f'\nTraining {model_name} model...')
    if model_name == 'GAT':
        # simplified GAT
        model = model_class(
            in_feats=graph.num_node_features,
            hidden_feats=32,   
            out_feats=num_classes,
            dropout=0.3     
        )
    else:
        
        if model_name == 'GCN':
            model = model_class(
                in_feats=graph.num_node_features,
                hidden_feats=128, 
                out_feats=num_classes,
                num_layers=2,
                dropout=0.5
            )
        elif model_name == 'GraphSAGE':
            model = model_class(
                in_feats=graph.num_node_features,
                hidden_feats=128,  
                out_feats=num_classes,
                num_layers=2,
                dropout=0.5
            )
        elif model_name == 'GraphGPS':
            model = model_class(
                in_feats=graph.num_node_features,
                hidden_feats=64,  # GraphGPS uses too much memory, so this needs to be reduced to not crash
                out_feats=num_classes,
                num_layers=2,
                num_heads=2,
                dropout=0.5
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    metrics = train_and_evaluate(
        model=model,
        graph=graph,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        device=device,
        num_classes=num_classes,
        epochs=10,           
        patience=3,          
        learning_rate=0.01,
        weight_decay=5e-4,
        dropout=0.5 if model_name != 'GAT' else 0.3
    )
    # Free up GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return metrics



def main():

    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    dataset = PygNodePropPredDataset(name='ogbn-mag', root='dataset/')
    data = dataset[0]
    data = data.to('cpu')

    
    print("Data keys:", list(data.keys()))
    print("Data has x_dict:", hasattr(data, 'x_dict'))
    print("Data has y_dict:", hasattr(data, 'y_dict'))

    
    try:
        x_paper = data.x_dict['paper']  # Shape: [num_paper_nodes, 128]
        y_paper = data.y_dict['paper'].squeeze(-1)  # Shape: [num_paper_nodes]
    except KeyError as e:
        print(f"KeyError accessing 'paper': {e}")
        print("Available keys in x_dict:", list(data.x_dict.keys))
        print("Available keys in y_dict:", list(data.y_dict.keys))
        return

    # Train/Val/Test
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']['paper']
    valid_idx = split_idx['valid']['paper']
    test_idx = split_idx['test']['paper']

    # Extract 'cites' edge index
    try:
        cites_edge_index = data.edge_index_dict[('paper', 'cites', 'paper')]  # Shape: [2, num_cites_edges]
    except KeyError as e:
        print(f"KeyError accessing ('paper', 'cites', 'paper') edges: {e}")
        print("Available edge types:", list(data.edge_index_dict.keys))
        return

    graph = Data(x=x_paper, edge_index=cites_edge_index, y=y_paper)
    graph = graph.to(device)
    train_idx = train_idx.to(device)
    valid_idx = valid_idx.to(device)
    test_idx = test_idx.to(device)

    num_classes = dataset.num_classes  # 349

    model_names = ['GCN', 'GraphSAGE', 'GraphGPS', 'GAT']
    model_classes = {
        'GCN': GCNModel,
        'GraphSAGE': GraphSAGEModel,
        'GraphGPS': GraphGPSModel,
        'GAT': GATModel
    }

    metrics = {}
    for model_name in model_names:
        metrics[model_name] = train_model(
            model_name=model_name,
            model_class=model_classes[model_name],
            graph=graph,
            train_idx=train_idx,
            val_idx=valid_idx,
            test_idx=test_idx,
            device=device,
            num_classes=num_classes
        )


    print("\n===== Final Metrics =====")
    for model_name, metric in metrics.items():
        print(f'\nModel: {model_name}')
        print(f"Validation Loss: {metric['val_loss']:.4f}")
        print(f"Validation Accuracy: {metric['val_acc']:.4f}")
        print(f"Validation Macro F1 Score: {metric['val_f1']:.4f}")
        print(f"Test Accuracy: {metric['test_acc']:.4f}")
        print(f"Test Macro F1 Score: {metric['test_f1']:.4f}")

if __name__ == "__main__":
    main()
