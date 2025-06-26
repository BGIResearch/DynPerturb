import torch
from torch import nn
from utils.utils import MergeLayer


# Temporal attention layer for temporal graph neural networks
class TemporalAttentionLayer(torch.nn.Module):
    """
    Temporal attention layer. Returns the temporal embedding of a node given the node itself,
    its neighbors, and the edge timestamps.
    """

    def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim, output_dimension, n_head=2, dropout=0.1):
        super(TemporalAttentionLayer, self).__init__()
        # Attention and feature dimensions
        self.n_head = n_head
        self.feat_dim = n_node_features
        self.time_dim = time_dim
        self.query_dim = n_node_features + time_dim
        self.key_dim = n_neighbors_features + time_dim + n_edge_features

        # Merger layer for combining attention output and node features
        self.merger = MergeLayer(self.query_dim, n_node_features, n_node_features, output_dimension)

        # Multi-head attention for temporal neighborhood
        self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim, kdim=self.key_dim, vdim=self.key_dim, num_heads=n_head, dropout=dropout)

    def forward(self, src_node_features, src_time_features, neighbors_features, neighbors_time_features, edge_features, neighbors_padding_mask):
        """
        Temporal attention model forward pass.
        :param src_node_features: Tensor [batch_size, n_node_features]
        :param src_time_features: Tensor [batch_size, 1, time_dim]
        :param neighbors_features: Tensor [batch_size, n_neighbors, n_node_features]
        :param neighbors_time_features: Tensor [batch_size, n_neighbors, time_dim]
        :param edge_features: Tensor [batch_size, n_neighbors, n_edge_features]
        :param neighbors_padding_mask: Bool Tensor [batch_size, n_neighbors]
        :return: attn_output [batch_size, output_dimension], attn_output_weights [batch_size, n_neighbors]
        """
        # Add time encoding to source node features
        src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)

        # Concatenate node and time features for query
        query = torch.cat([src_node_features_unrolled, src_time_features], dim=2)

        # Concatenate neighbor, edge, and time features for key/value
        key = torch.cat([neighbors_features, edge_features, neighbors_time_features], dim=2)

        # Reshape tensors to expected shape for multi-head attention
        query = query.permute([1, 0, 2])  # [1, batch_size, query_dim]
        key = key.permute([1, 0, 2])      # [n_neighbors, batch_size, key_dim]

        # Compute mask for source nodes with no valid neighbors
        invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
        
        # If a source node has no valid neighbor, set its first neighbor to be valid
        neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False

        # Compute multi-head attention output and weights
        attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=key, key_padding_mask=neighbors_padding_mask)
        attn_output = attn_output.squeeze()
        attn_output_weights = attn_output_weights.squeeze()

        # Set attention output to zero for nodes with no valid neighbors
        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
        attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)

        # Skip connection: merge attention output and original node features
        attn_output = self.merger(attn_output, src_node_features)
        return attn_output, attn_output_weights
