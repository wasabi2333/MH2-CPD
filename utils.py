import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

def set_seed(seed=42):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_directories(config):
    
    directories = [
        config.results_save_path,
        os.path.join(config.results_save_path, 'plots'),
        os.path.join(config.results_save_path, 'models'),
        os.path.join(config.results_save_path, 'reports')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_config(config, save_path):
    
    config_dict = {
        attr: getattr(config, attr) 
        for attr in dir(config) 
        if not attr.startswith('_')
    }
    
    
    for key, value in config_dict.items():
        if isinstance(value, torch.device):
            config_dict[key] = str(value)
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

def plot_training_history(train_losses, test_losses, test_accuracies, save_path=None):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    
    ax2.plot(epochs, test_accuracies, 'g-', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_hypergraph_statistics(stats, save_path=None):
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    
    ax = axes[0, 0]
    ax.hist(stats['node_degrees'], bins=20, edgecolor='black')
    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Node Degree Distribution (Mean: {stats["avg_node_degree"]:.2f})')
    
    
    ax = axes[0, 1]
    ax.hist(stats['edge_degrees'], bins=20, edgecolor='black')
    ax.set_xlabel('Hyperedge Degree')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Hyperedge Degree Distribution (Mean: {stats["avg_edge_degree"]:.2f})')
    
    
    ax = axes[1, 0]
    if 'layer_statistics' in stats:
        
        layers = ['Layer 1: Group', 'Layer 2: Intra-group', 'Layer 3: Inter-group']
        counts = [
            stats['layer_statistics']['group'],
            stats['layer_statistics']['intra_group_semantic'],
            stats['layer_statistics']['inter_group']
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        ax.bar(layers, counts, color=colors)
        ax.set_xlabel('Hyperedge Layer')
        ax.set_ylabel('Count')
        ax.set_title('Three-layer Hyperedge Distribution')
        
        
        for i, count in enumerate(counts):
            ax.text(i, count + 0.5, str(count), ha='center')
    else:
        
        edge_types = list(stats['edge_types'].keys())
        edge_counts = list(stats['edge_types'].values())
        ax.bar(edge_types, edge_counts)
        ax.set_xlabel('Hyperedge Type')
        ax.set_ylabel('Count')
        ax.set_title('Hyperedge Type Distribution')
    
    
    ax = axes[1, 1]
    ax.axis('off')
    info_text = f"Hypergraph Statistics\n\n"
    info_text += f"Total Nodes: {stats['n_nodes']}\n"
    info_text += f"Total Hyperedges: {stats['n_edges']}\n"
    info_text += f"Average Node Degree: {stats['avg_node_degree']:.2f}\n"
    info_text += f"Average Hyperedge Degree: {stats['avg_edge_degree']:.2f}\n"
    ax.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_experiment_results(config, train_losses, test_losses, test_accuracies, 
                          test_results, hypergraph_stats):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_results = {
        'timestamp': timestamp,
        'config': {
            attr: getattr(config, attr) 
            for attr in dir(config) 
            if not attr.startswith('_') and not isinstance(getattr(config, attr), torch.device)
        },
        'training_history': {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies
        },
        'test_results': {
            'accuracy': float(test_results['accuracy']),
            'precision': float(test_results['precision']),
            'recall': float(test_results['recall']),
            'f1': float(test_results['f1']),
            'auc': float(test_results['auc'])
        },
        'hypergraph_stats': {
            'n_nodes': int(hypergraph_stats['n_nodes']),
            'n_edges': int(hypergraph_stats['n_edges']),
            'avg_node_degree': float(hypergraph_stats['avg_node_degree']),
            'avg_edge_degree': float(hypergraph_stats['avg_edge_degree']),
            'edge_types': hypergraph_stats['edge_types']
        }
    }
    
    save_path = os.path.join(config.results_save_path, f'experiment_results_{timestamp}.json')
    with open(save_path, 'w') as f:
        json.dump(experiment_results, f, indent=4)
    
    print(f"Experiment results saved to: {save_path}")
    
    return save_path

def print_model_summary(model, input_shape, H_shape):
    """Print model summary"""
    print("\n" + "="*60)
    print("Model Architecture Summary")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input shape: {input_shape}")
    print(f"Hypergraph shape: {H_shape}")
    print("="*60 + "\n")