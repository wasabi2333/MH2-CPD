# Construction Project Delay Risk Prediction - Hypergraph GNN

This is a binary classification system using Hypergraph Neural Network (Hypergraph GNN) to predict construction project delay risk. The model uses 37 risk factors (grouped into 7 categories) to predict a single project outcome indicator (delay) as either low risk or high risk.

## Project Features
- **Binary classification**: Risk classification based on standardized values 
- **Hypergraph structure**: Three-layer hyperedge construction strategy
  - Layer 1: Domain knowledge group hyperedges
  - Layer 2: Intra-group semantic hyperedges (based on mutual information)
  - Layer 3: Inter-group hyperedges (based on clustering strategy)
- **Attention mechanism**: Attention mechanism in hypergraph convolution

## Environment Requirements

```bash
Python >= 3.7
PyTorch >= 1.8.0
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
tqdm
```

