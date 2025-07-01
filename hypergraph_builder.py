import numpy as np
import torch
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr

class HypergraphBuilder:
    
    
    def __init__(self, feature_cols, factor_groups):
        
        self.feature_cols = feature_cols
        self.factor_groups = factor_groups
        self.n_nodes = len(feature_cols)
        
        
        self.feature_to_group = {}
        for group_name, features in factor_groups.items():
            for feature in features:
                self.feature_to_group[feature] = group_name
    
    def build_group_hyperedges(self):
        
        hyperedges = []
        
        for group_name, group_features in self.factor_groups.items():
            edge = [self.feature_cols.index(f) for f in group_features]
            hyperedges.append({
                'nodes': edge,
                'type': 'group',
                'name': group_name
            })
            
        return hyperedges
    
    def build_intra_group_semantic_hyperedges(self, X, mi_threshold=0.3, max_edge_size=4):
        
        hyperedges = []
        
        
        for group_name, group_features in self.factor_groups.items():
            group_indices = [self.feature_cols.index(f) for f in group_features]
            
            if len(group_indices) < 2:
                continue
            
            
            n_group = len(group_indices)
            mi_matrix = np.zeros((n_group, n_group))
            
            for i in range(n_group):
                for j in range(i+1, n_group):
                    
                    x_i = self._discretize(X[:, group_indices[i]])
                    x_j = self._discretize(X[:, group_indices[j]])
                    
                    
                    mi = mutual_info_score(x_i, x_j)
                    
                    h_i = mutual_info_score(x_i, x_i)
                    h_j = mutual_info_score(x_j, x_j)
                    if min(h_i, h_j) > 0:
                        mi_normalized = mi / min(h_i, h_j)
                    else:
                        mi_normalized = 0
                    
                    mi_matrix[i, j] = mi_normalized
                    mi_matrix[j, i] = mi_normalized
            
            
            visited = set()
            
            for i in range(n_group):
                if i in visited:
                    continue
                
                
                high_mi_features = [i]
                mi_values = []
                
                for j in range(n_group):
                    if i != j and mi_matrix[i, j] > mi_threshold:
                        high_mi_features.append(j)
                        mi_values.append(mi_matrix[i, j])
                
                
                if len(high_mi_features) >= 2:
                    
                    if len(high_mi_features) > max_edge_size:
                        sorted_indices = np.argsort(mi_values)[::-1][:max_edge_size-1]
                        high_mi_features = [i] + [high_mi_features[idx+1] for idx in sorted_indices]
                    
                    
                    global_indices = [group_indices[idx] for idx in high_mi_features]
                    
                    
                    avg_mi = np.mean([
                        mi_matrix[idx1, idx2] 
                        for pos1, idx1 in enumerate(high_mi_features) 
                        for pos2, idx2 in enumerate(high_mi_features) 
                        if pos1 < pos2
                    ])
                    
                    hyperedges.append({
                        'nodes': global_indices,
                        'type': 'intra_group_semantic',
                        'group': group_name,
                        'avg_mutual_info': avg_mi
                    })
                    
                    visited.update(high_mi_features)
        
        return hyperedges
    
    def build_inter_group_hyperedges(self, X, n_clusters=5, min_groups_per_edge=2):
        
        hyperedges = []
        
        
        similarity_matrix = self._compute_similarity_matrix(X)
        
        
        clustering = SpectralClustering(
            n_clusters=n_clusters, 
            affinity='precomputed',
            random_state=42
        )
        cluster_labels = clustering.fit_predict(similarity_matrix)
        
        
        for cluster_id in range(n_clusters):
            nodes = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            
            groups_in_cluster = set()
            for node_idx in nodes:
                feature_name = self.feature_cols[node_idx]
                groups_in_cluster.add(self.feature_to_group[feature_name])
            
            
            if len(groups_in_cluster) >= min_groups_per_edge and 2 <= len(nodes) <= 8:
                hyperedges.append({
                    'nodes': nodes,
                    'type': 'inter_group',
                    'cluster_id': cluster_id,
                    'groups': list(groups_in_cluster),
                    'n_groups': len(groups_in_cluster)
                })
        
        
        if len(hyperedges) < 3:
            hyperedges.extend(self._hierarchical_inter_group_clustering(X, similarity_matrix))
        
        return hyperedges
    
    def _hierarchical_inter_group_clustering(self, X, similarity_matrix):
        
        hyperedges = []
        
        
        distance_matrix = 1 - similarity_matrix
        
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.5,
            metric='precomputed',
            linkage='average'
        )
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        
        unique_labels = np.unique(cluster_labels)
        for cluster_id in unique_labels:
            nodes = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            
            groups_in_cluster = set()
            for node_idx in nodes:
                feature_name = self.feature_cols[node_idx]
                groups_in_cluster.add(self.feature_to_group[feature_name])
            
            if len(groups_in_cluster) >= 2 and 3 <= len(nodes) <= 6:
                hyperedges.append({
                    'nodes': nodes,
                    'type': 'inter_group_hierarchical',
                    'groups': list(groups_in_cluster)
                })
        
        return hyperedges
    
    def build_hypergraph(self, X, mi_threshold=0.3, n_clusters=5):
        
        all_hyperedges = []
        
        # Layer 1: Group hyperedges
        print("Building Layer 1: Domain knowledge group hyperedges...")
        group_edges = self.build_group_hyperedges()
        all_hyperedges.extend(group_edges)
        print(f"  Created {len(group_edges)} group hyperedges")
        
        # Layer 2: Intra-group semantic hyperedges
        print("Building Layer 2: Intra-group semantic hyperedges (based on mutual information)...")
        semantic_edges = self.build_intra_group_semantic_hyperedges(X, mi_threshold)
        all_hyperedges.extend(semantic_edges)
        print(f"  Created {len(semantic_edges)} intra-group semantic hyperedges")
        
        # Layer 3: Inter-group hyperedges
        print("Building Layer 3: Inter-group hyperedges (based on clustering)...")
        inter_group_edges = self.build_inter_group_hyperedges(X, n_clusters)
        all_hyperedges.extend(inter_group_edges)
        print(f"  Created {len(inter_group_edges)} inter-group hyperedges")
        
        # Build incidence matrix
        n_edges = len(all_hyperedges)
        incidence_matrix = np.zeros((self.n_nodes, n_edges))
        
        for edge_idx, edge_info in enumerate(all_hyperedges):
            for node_idx in edge_info['nodes']:
                incidence_matrix[node_idx, edge_idx] = 1
        
        
        print(f"\nHypergraph construction completed:")
        print(f"  Number of nodes: {self.n_nodes}")
        print(f"  Number of hyperedges: {n_edges}")
        print(f"    - Group hyperedges: {len(group_edges)}")
        print(f"    - Intra-group semantic hyperedges: {len(semantic_edges)}")
        print(f"    - Inter-group hyperedges: {len(inter_group_edges)}")
        
        
        node_degrees = incidence_matrix.sum(axis=1)
        print(f"  Average node degree: {node_degrees.mean():.2f}")
        print(f"  Min/Max node degree: {node_degrees.min():.0f}/{node_degrees.max():.0f}")
        
        return torch.FloatTensor(incidence_matrix), all_hyperedges
    
    def _discretize(self, x, n_bins=10):
        """Discretize continuous values for mutual information calculation"""
        
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(x, percentiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        
        bins = np.unique(bins)
        
        return np.digitize(x, bins[1:-1])
    
    def _compute_similarity_matrix(self, X):
        
        n_features = X.shape[1]
        similarity_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i, n_features):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    
                    corr = abs(np.corrcoef(X[:, i], X[:, j])[0, 1])
                    
                    
                    x_i = self._discretize(X[:, i])
                    x_j = self._discretize(X[:, j])
                    mi = mutual_info_score(x_i, x_j)
                    h_i = mutual_info_score(x_i, x_i)
                    h_j = mutual_info_score(x_j, x_j)
                    if min(h_i, h_j) > 0:
                        nmi = mi / min(h_i, h_j)
                    else:
                        nmi = 0
                    
                    
                    norm_i = np.linalg.norm(X[:, i])
                    norm_j = np.linalg.norm(X[:, j])
                    if norm_i > 0 and norm_j > 0:
                        cos_sim = np.dot(X[:, i], X[:, j]) / (norm_i * norm_j)
                        cos_sim = (cos_sim + 1) / 2  
                    else:
                        cos_sim = 0
                    
                    
                    similarity = 0.4 * corr + 0.4 * nmi + 0.2 * cos_sim
                    
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def get_hypergraph_statistics(self, incidence_matrix, hyperedge_info):
        
        H = incidence_matrix.numpy() if torch.is_tensor(incidence_matrix) else incidence_matrix
        
        stats = {
            'n_nodes': H.shape[0],
            'n_edges': H.shape[1],
            'node_degrees': H.sum(axis=1),
            'edge_degrees': H.sum(axis=0),
            'avg_node_degree': H.sum(axis=1).mean(),
            'avg_edge_degree': H.sum(axis=0).mean(),
            'edge_types': {},
            'layer_statistics': {
                'group': 0,
                'intra_group_semantic': 0,
                'inter_group': 0
            }
        }
        
        
        for edge in hyperedge_info:
            edge_type = edge['type']
            if edge_type not in stats['edge_types']:
                stats['edge_types'][edge_type] = 0
            stats['edge_types'][edge_type] += 1
            
            
            if edge_type == 'group':
                stats['layer_statistics']['group'] += 1
            elif edge_type == 'intra_group_semantic':
                stats['layer_statistics']['intra_group_semantic'] += 1
            elif edge_type in ['inter_group', 'inter_group_hierarchical']:
                stats['layer_statistics']['inter_group'] += 1
        
        return stats