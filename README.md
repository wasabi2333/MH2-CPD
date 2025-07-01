# Multi-view Heterogeneous Hypergraph Neural Network for Construction Project Delay Prediction (MH2-CPD)

MH2-CPD is an advanced neural network framework for delay prediction in construction projects. Unlike traditional graph-based models, MH2-CPD leverages a multiview heterogeneous hypergraph structure to capture complex, high-order, and multi-relational dependencies among diverse project features.

The framework consists of two main components:
1. **Multi-view Heterogeneous Hypergraph Construction**: Multiple types of hyperedges are constructed based on feature groups, correlations, and latent patterns, enabling the model to represent heterogeneous and high-order relationships among project attributes from different perspectives.
2. **Hypergraph Neural Network with Attention**: A multi-layer neural architecture performs hypergraph convolution operations [1], enhanced by layer normalisation, residual connections, and an edge attention mechanism, to jointly predict delay risk in a multi-task learning setting.

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

## Citation
```
@misc{mh2-cpd2025,
    title={Multi-view Heterogeneous Hypergraph Neural Network for Construction Project Delay Prediction},
    author={[Authors]},
    year={2025},
    note={Manuscript in preparation}
}
```
## References

[1] Feng, Y., You, H., Zhang, Z., Ji, R., & Gao, Y. (2019). Hypergraph neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 3558-3565).
