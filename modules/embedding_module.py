import torch
from torch import nn
import numpy as np
import math
import os
import json
from model.temporal_attention import TemporalAttentionLayer
from collections import defaultdict

# EmbeddingModule: Base class for node embedding modules
class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers, n_node_features, n_edge_features, n_time_features, embedding_dimension, device, dropout):
        super(EmbeddingModule, self).__init__()
        
        # Store feature and module references
        self.node_features = node_features
        self.edge_features = edge_features
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device
        
        # Buffer for saving embeddings: {node_id: [(timestamp, embedding), ...]}
        self.saved_embeddings = defaultdict(list)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None, use_time_proj=True):
        
        # To be implemented by subclasses
        return NotImplemented

    def save_embeddings(self, save_dir, epoch):
        """
        Save collected embeddings to disk and clear the buffer.
        :param save_dir: Directory to save embeddings.
        :param epoch: Current epoch number.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        serializable_embeddings = {}
        for node, emb_list in self.saved_embeddings.items():
            serializable_embeddings[int(node)] = [{"timestamp": float(ts), "embedding": emb.tolist()} for ts, emb in emb_list]
            
        save_path = os.path.join(save_dir, f"embeddings_epoch_{epoch}.json")
        with open(save_path, "w") as f:
            json.dump(serializable_embeddings, f)
        print(f"Saved embeddings for epoch {epoch} at {save_path}")
        
        self.saved_embeddings = defaultdict(list)

    def save_embeddings_new(self, save_dir, node_to_modify, modify_time):
        """
        Save embeddings for a specific node/time to disk and clear the buffer.
        :param save_dir: Directory to save embeddings.
        :param node_to_modify: Node id.
        :param modify_time: Time value.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        serializable_embeddings = {}
        for node, emb_list in self.saved_embeddings.items():
            serializable_embeddings[int(node)] = [{"timestamp": float(ts), "embedding": emb.tolist()} for ts, emb in emb_list]
            
        save_path = os.path.join(save_dir, f"embeddings_{node_to_modify}_{modify_time}.json")
        with open(save_path, "w") as f:
            json.dump(serializable_embeddings, f)
        print(f"Saved embeddings new at {save_path}")
        
        self.saved_embeddings = defaultdict(list)

    def log_embedding(self, node_ids, timestamps, embeddings):
        """
        Log embeddings to the buffer for later saving.
        :param node_ids: List or tensor of node IDs.
        :param timestamps: List or tensor of timestamps.
        :param embeddings: Tensor of embeddings [batch_size, embedding_dim].
        """
        for node_id, ts, emb in zip(node_ids, timestamps, embeddings):
            node_id = int(node_id)
            ts = float(ts.item())
            emb = emb.detach().cpu()
            
            # Check if timestamp already exists for this node
            existing = next(((i, e) for i, (time, e) in enumerate(self.saved_embeddings[node_id]) if time == ts), None)
            if existing:
                index, _ = existing
                self.saved_embeddings[node_id][index] = (ts, emb)
            else:
                self.saved_embeddings[node_id].append((ts, emb))

    def get_node_features_at_time(self, node_ids, timestamps):
        """
        Get node features at given timestamps. If not available, use the closest or zero vector.
        :param node_ids: List or Tensor of node IDs (batch)
        :param timestamps: List or Tensor of corresponding timestamps (batch)
        :return: Tensor [batch_size, embedding_dim]
        """
        batch_features = []
        # Iterate over each node and timestamp in the batch
        for node_id, timestamp in zip(node_ids, timestamps):
            node_id = int(node_id.item())
            timestamp = float(timestamp.item())
            
            # Check if node has features for the given timestamp
            if node_id in self.node_features:
                node_time_features = self.node_features[node_id]
                if timestamp in node_time_features:
                    feature_vector = node_time_features[timestamp]
                else:
                    # Use zero vector if timestamp not found
                    feature_vector = np.zeros(self.n_node_features, dtype=np.float32)
            else:
                # Use zero vector if node not found
                feature_vector = np.zeros(self.n_node_features, dtype=np.float32)
                
            feature_vector = torch.tensor(feature_vector, dtype=torch.float32, device=self.device)
            batch_features.append(feature_vector)
            
        # Stack all feature vectors into a batch tensor
        return torch.stack(batch_features)

# IdentityEmbedding: Returns memory as embedding
class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None, use_time_proj=True):
        # Directly use memory as embedding
        embeddings = memory[source_nodes, :]
        self.log_embedding(source_nodes, timestamps, embeddings)
        return embeddings

# TimeEmbedding: Embedding with time-dependent scaling
class TimeEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
        n_node_features, n_edge_features, n_time_features, embedding_dimension, device, n_heads=2,
        dropout=0.1, use_memory=True, n_neighbors=1):
        
        super(TimeEmbedding, self).__init__(node_features, edge_features, memory, neighbor_finder,
            time_encoder, n_layers, n_node_features, n_edge_features, n_time_features, embedding_dimension, device, dropout)
        
        # Linear layer with normal initialization for time scaling
        class NormalLinear(nn.Linear):
            def reset_parameters(self):
                stdv = 1.0 / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)
                    
        self.embedding_layer = NormalLinear(1, self.n_node_features)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None, use_time_proj=True):
        # Time-dependent scaling of memory
        source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))
        self.log_embedding(source_nodes, timestamps, source_embeddings)
        
        return source_embeddings

# GraphEmbedding: Recursive temporal graph embedding base class
class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers, n_node_features,
        n_edge_features, n_time_features, embedding_dimension, device, n_heads=2, dropout=0.1, use_memory=True):
        super(GraphEmbedding, self).__init__(node_features, edge_features, memory, neighbor_finder, time_encoder,
            n_layers, n_node_features, n_edge_features, n_time_features, embedding_dimension, device, dropout)
        
        self.use_memory = use_memory
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, node_features, n_layers, n_neighbors=20, time_diffs=None, use_time_proj=True):
        """Recursive computation of temporal graph attention layers."""
        assert n_layers >= 0
        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)
        source_node_features = self.get_node_features_at_time(source_nodes_torch, timestamps_torch)
        
        if self.use_memory:
            source_node_features = memory[source_nodes, :] + source_node_features
            
        if n_layers == 0:
            embeddings = source_node_features
            self.log_embedding(source_nodes, timestamps, embeddings)
            
            return embeddings
        else:
            # Recursive neighbor aggregation
            neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes, timestamps, n_neighbors=n_neighbors)
            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

            # Compute edge time differences (delta)
            edge_deltas = timestamps[:, np.newaxis] - edge_times
            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            # Get features for neighbor nodes at the given timestamps
            neighbor_node_features = self.get_node_features_at_time(neighbors.flatten(), np.repeat(timestamps, n_neighbors))
            
            # Recursively compute neighbor embeddings (one layer less)
            neighbor_embeddings = self.compute_embedding(memory, neighbors.flatten(), np.repeat(timestamps, n_neighbors), neighbor_node_features, n_layers=n_layers - 1, n_neighbors=n_neighbors)
            
            # Reshape neighbor embeddings for aggregation
            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)

            # Encode edge time differences and source node time
            edge_time_embeddings = self.time_encoder(edge_deltas_torch)
            source_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))

            # Get edge features for the neighbor edges
            edge_features = self.edge_features[edge_idxs, :]

            # Mask for padding (if neighbor id is 0)
            mask = neighbors_torch == 0
            
            # Aggregate all information to get source node embedding
            source_embedding = self.aggregate(n_layers, source_node_features, source_nodes_time_embedding, neighbor_embeddings, edge_time_embeddings, edge_features, mask)
            self.log_embedding(source_nodes, timestamps, source_embedding)
            
        return source_embedding

    def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding, neighbor_embeddings, edge_time_embeddings, edge_features, mask):
        # To be implemented by subclasses
        return NotImplemented

# GraphSumEmbedding: Aggregates neighbor embeddings by sum
class GraphSumEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
        n_node_features, n_edge_features, n_time_features, embedding_dimension, device, n_heads=2, dropout=0.1, use_memory=True):
        
        super(GraphSumEmbedding, self).__init__(node_features=node_features, edge_features=edge_features, memory=memory,
            neighbor_finder=neighbor_finder, time_encoder=time_encoder, n_layers=n_layers, n_node_features=n_node_features,
            n_edge_features=n_edge_features, n_time_features=n_time_features, embedding_dimension=embedding_dimension,
            device=device, n_heads=n_heads, dropout=dropout, use_memory=use_memory)
        
        # Linear layers for neighbor and source aggregation
        self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features + n_edge_features, embedding_dimension,) for _ in range(n_layers)])
        self.linear_2 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_node_features + n_time_features, embedding_dimension,) for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding, neighbor_embeddings, edge_time_embeddings, edge_features, mask):
        # Aggregate neighbor embeddings and combine with source features
        # Concatenate neighbor embeddings, edge time embeddings, and edge features
        neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features], dim=2)

        # Transform neighbor features with linear layer
        neighbor_embeddings_transformed = self.linear_1[n_layer - 1](neighbors_features)

        # Sum over all neighbors and apply ReLU
        neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings_transformed, dim=1))

        # Concatenate source node features and time embedding
        source_features = torch.cat([source_node_features, source_nodes_time_embedding.squeeze()], dim=1)

        # Combine neighbor sum and source features, then transform
        source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
        source_embedding = self.linear_2[n_layer - 1](source_embedding)
        
        return source_embedding

# GraphAttentionEmbedding: Uses temporal attention for aggregation
class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers, n_node_features,
        n_edge_features, n_time_features, embedding_dimension, device, n_heads=2, dropout=0.1, use_memory=True):
        
        super(GraphAttentionEmbedding, self).__init__(node_features=node_features, edge_features=edge_features,
            memory=memory, neighbor_finder=neighbor_finder, time_encoder=time_encoder, n_layers=n_layers,
            n_node_features=n_node_features, n_edge_features=n_edge_features, n_time_features=n_time_features,
            embedding_dimension=embedding_dimension, device=device, n_heads=n_heads, dropout=dropout, use_memory=use_memory)
        
        # Temporal attention layers for aggregation
        self.attention_models = torch.nn.ModuleList([
            TemporalAttentionLayer(n_node_features=n_node_features, n_neighbors_features=n_node_features,
                                n_edge_features=n_edge_features, time_dim=n_time_features,
                                output_dimension=n_node_features, n_head=n_heads, dropout=dropout) for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding, neighbor_embeddings, edge_time_embeddings, edge_features, mask):
        # Apply temporal attention for aggregation
        attention_model = self.attention_models[n_layer - 1]
        attn_output, attn_output_weights = attention_model(source_node_features, source_nodes_time_embedding, neighbor_embeddings, edge_time_embeddings, edge_features, mask)
        
        return attn_output

# Factory function for embedding module selection
def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                        time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                        embedding_dimension, device, n_heads=2, dropout=0.1, n_neighbors=None, use_memory=True):
    
    if module_type == "graph_attention":
        return GraphAttentionEmbedding(node_features=node_features, edge_features=edge_features, memory=memory,
                                    neighbor_finder=neighbor_finder, time_encoder=time_encoder, n_layers=n_layers,
                                    n_node_features=n_node_features, n_edge_features=n_edge_features,
                                    n_time_features=n_time_features, embedding_dimension=embedding_dimension,
                                    device=device, n_heads=n_heads, dropout=dropout, use_memory=use_memory)
        
    elif module_type == "graph_sum":
        return GraphSumEmbedding(node_features=node_features, edge_features=edge_features, memory=memory,
                                neighbor_finder=neighbor_finder, time_encoder=time_encoder, n_layers=n_layers,
                                n_node_features=n_node_features, n_edge_features=n_edge_features,
                                n_time_features=n_time_features, embedding_dimension=embedding_dimension,
                                device=device, n_heads=n_heads, dropout=dropout, use_memory=use_memory)
        
    elif module_type == "identity":
        return IdentityEmbedding(node_features=node_features, edge_features=edge_features, memory=memory,
                                neighbor_finder=neighbor_finder, time_encoder=time_encoder,
                                n_layers=n_layers, n_node_features=n_node_features,
                                n_edge_features=n_edge_features, n_time_features=n_time_features,
                                embedding_dimension=embedding_dimension, device=device, dropout=dropout)
        
    elif module_type == "time":
        return TimeEmbedding(node_features=node_features, edge_features=edge_features, memory=memory,
                            neighbor_finder=neighbor_finder, time_encoder=time_encoder, n_layers=n_layers,
                            n_node_features=n_node_features, n_edge_features=n_edge_features,
                            n_time_features=n_time_features, embedding_dimension=embedding_dimension,
                            device=device, dropout=dropout, n_neighbors=n_neighbors)
        
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))
