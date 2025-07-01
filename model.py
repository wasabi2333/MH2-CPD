import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperGraphConv(nn.Module):
    
    
    def __init__(self, in_features, out_features, use_attention=True):
        
        super(HyperGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_attention = use_attention
        
        
        self.linear = nn.Linear(in_features, out_features)
        
        
        if use_attention:
            self.attention_node = nn.Linear(out_features, 1)
            self.attention_edge = nn.Linear(out_features, 1)
        
        
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, H):
        
        batch_size = x.size(0)
        n_nodes = x.size(1)
        n_edges = H.size(1)
        
        
        x = self.linear(x)  
        
        if self.use_attention:
            
            node_scores = self.attention_node(x)  
            node_scores = torch.sigmoid(node_scores)
            x = x * node_scores
        
        
        edge_features = []
        for b in range(batch_size):
            
            edge_feat = torch.matmul(H.T, x[b])  
            edge_features.append(edge_feat)
        
        edge_features = torch.stack(edge_features)  
        
        
        edge_degree = H.sum(dim=0).clamp(min=1)  
        edge_features = edge_features / edge_degree.unsqueeze(0).unsqueeze(-1)
        
        if self.use_attention:
            
            edge_scores = self.attention_edge(edge_features)  
            edge_scores = torch.sigmoid(edge_scores)
            edge_features = edge_features * edge_scores
        
        
        out = []
        for b in range(batch_size):
            
            node_feat = torch.matmul(H, edge_features[b]) 
            out.append(node_feat)
        
        out = torch.stack(out)  
        
        
        node_degree = H.sum(dim=1).clamp(min=1) 
        out = out / node_degree.unsqueeze(0).unsqueeze(-1)
        
        
        out_reshaped = out.view(-1, self.out_features)
        out_reshaped = self.bn(out_reshaped)
        out = out_reshaped.view(batch_size, n_nodes, self.out_features)
        
        out = F.relu(out)
        out = self.dropout(out)
        
        return out

class HyperGNN(nn.Module):
    
    
    def __init__(self, input_dim, hidden_dim, num_layers=3, num_classes=2):
        
        super(HyperGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        
        self.input_proj = nn.Linear(1, hidden_dim)
        
        
        self.hgc_layers = nn.ModuleList()
        for i in range(num_layers):
            self.hgc_layers.append(
                HyperGraphConv(hidden_dim, hidden_dim, use_attention=True)
            )
        
        
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x, H):
        
        batch_size = x.size(0)
        n_features = x.size(1)
        
        
        x = x.unsqueeze(2)  
        x = self.input_proj(x)  
        
        
        for hgc in self.hgc_layers:
            x_new = hgc(x, H)
            x = x + x_new  
        
        
        x = torch.mean(x, dim=1)  
        x = self.global_pool(x)
        
        
        output = self.classifier(x)  
        
        return output