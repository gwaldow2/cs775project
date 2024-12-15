import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_CSV = 'benchmark_results.csv'
OUTPUT_DIR = 'plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)
model_colors = {
    'GCN': 'blue',
    'GraphSAGE': 'orange',
    'GraphGPS': 'green',
    'GAT': 'red'
}

df = pd.read_csv(RESULTS_CSV)

datasets = [
    {'name': 'ogbn-arxiv', 'metric': 'val_f1_macro', 'ylabel': 'Val F1 Macro'},
    {'name': 'ogbn-products', 'metric': 'val_f1_macro', 'ylabel': 'Val F1 Macro'},
    {'name': 'ogbn-proteins', 'metric': 'test_rocauc', 'ylabel': 'Test ROC AUC'}
]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18))

for ax, dataset in zip(axes, datasets):
    dataset_name = dataset['name']
    metric = dataset['metric']
    ylabel = dataset['ylabel']
    
    dataset_df = df[df['dataset'] == dataset_name]
    if dataset_df.empty:
        print(f"No data found for dataset '{dataset_name}'. Skipping plot.")
        continue
    
    latest_epochs = dataset_df.groupby('model')['epoch'].max().reset_index()
    merged = pd.merge(latest_epochs, dataset_df, on=['model', 'epoch'], how='left')
    metric_df = merged[['model', metric]].dropna()
    
    if metric_df.empty:
        print(f"No valid data for metric '{metric}' in dataset '{dataset_name}'. Skipping plot.")
        continue
    
    metric_df_sorted = metric_df.sort_values(by=metric, ascending=False)
    palette = [model_colors.get(model, 'gray') for model in metric_df_sorted['model']]
    sns.barplot(x='model', y=metric, data=metric_df_sorted, palette=palette, ax=ax)
    
    # annotate
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.01,  # Fixed y-offset
                f"{height:.3f}",
                ha='center', va='bottom', fontsize=12, color='black')
    
    ax.set_title(f'Performance on {dataset_name}', fontsize=16)
    ax.set_xlabel('GNN Method', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
plt.tight_layout()

# collage
collage_path = os.path.join(OUTPUT_DIR, 'performance_collage.png')
plt.savefig(collage_path, bbox_inches='tight')
plt.close()

print(f"Collage plot has been saved to '{collage_path}'.")
