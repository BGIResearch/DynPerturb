import torch
from torch import nn
import numpy as np
import math
import os
import json

from model.temporal_attention import TemporalAttentionLayer
from collections import defaultdict


class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 dropout):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        # self.memory = memory
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device

        # Initialize a buffer to store embeddings
        self.saved_embeddings = defaultdict(list)  # {node_id: [(timestamp, embedding), ...]}

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                         use_time_proj=True):
        return NotImplemented

    def save_embeddings(self, save_dir, epoch):
        """
        Saves the collected embeddings to disk and clears the buffer.
        :param save_dir: Directory where embeddings will be saved.
        :param epoch: Current epoch number.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Prepare a serializable structure
        serializable_embeddings = {}
        for node, emb_list in self.saved_embeddings.items():
            serializable_embeddings[int(node)] = [
                {"timestamp": float(ts), "embedding": emb.tolist()} for ts, emb in emb_list
            ]

        # Save as JSON
        save_path = os.path.join(save_dir, f"embeddings_epoch_{epoch}.json")
        with open(save_path, 'w') as f:
            json.dump(serializable_embeddings, f)

        print(f"Saved embeddings for epoch {epoch} at {save_path}")

        # Clear the buffer after saving
        self.saved_embeddings = defaultdict(list)
        
    def save_embeddings_new(self, save_dir,node_to_modify,modify_time):
        """
        Saves the collected embeddings to disk and clears the buffer.
        :param save_dir: Directory where embeddings will be saved.
        :param epoch: Current epoch number.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Prepare a serializable structure
        serializable_embeddings = {}
        for node, emb_list in self.saved_embeddings.items():
            serializable_embeddings[int(node)] = [
                {"timestamp": float(ts), "embedding": emb.tolist()} for ts, emb in emb_list
            ]

        # Save as JSON
        save_path = os.path.join(save_dir, f"embeddings_{node_to_modify}_{modify_time}.json")
        with open(save_path, 'w') as f:
            json.dump(serializable_embeddings, f)

        print(f"Saved embeddings new at {save_path}")

        # Clear the buffer after saving
        self.saved_embeddings = defaultdict(list)

    def log_embedding(self, node_ids, timestamps, embeddings):
        """
        Logs the embeddings to the buffer.
        :param node_ids: List or tensor of node IDs.
        :param timestamps: List or tensor of timestamps.
        :param embeddings: Tensor of embeddings [batch_size, embedding_dim].
        """
        for node_id, ts, emb in zip(node_ids, timestamps, embeddings):
            node_id = int(node_id)
            ts = float(ts.item())
            emb = emb.detach().cpu()
            # 查找是否已经存在相同的时间戳
            existing = next(((i, e) for i, (time, e) in enumerate(self.saved_embeddings[node_id]) if time == ts), None)
            if existing:
                index, _ = existing
                self.saved_embeddings[node_id][index] = (ts, emb)
            else:
                self.saved_embeddings[node_id].append((ts, emb))

    def get_node_features_at_time(self, node_ids, timestamps):
        """
        获取每个节点在给定时间戳的特征，如果该时间戳不存在，则使用最近的时间戳的特征
        :param node_ids: List or Tensor of node IDs (batch)
        :param timestamps: List or Tensor of corresponding timestamps (batch)
        :return: Tensor [batch_size, embedding_dim]
        """
        batch_features = []
        
        for node_id, timestamp in zip(node_ids, timestamps):
            node_id = int(node_id.item())  # 确保 node_id 是整数
            timestamp = float(timestamp.item())  # 确保时间戳是浮点数
            
            # 1️⃣ 先确保该节点在字典中
            if node_id in self.node_features:
                node_time_features = self.node_features[node_id]  # 获取该节点的时间特征字典
                
                # 2️⃣ 如果该时间戳存在，直接取该时间戳特征
                if timestamp in node_time_features:
                    feature_vector = node_time_features[timestamp]
                
                # 3️⃣ 如果该时间戳不存在，找到最近的时间戳
                else:
                    feature_vector = np.zeros(self.n_node_features, dtype=np.float32)
            
            else:
                # 4️⃣ 该节点没有任何特征，返回零向量
                feature_vector = np.zeros(self.n_node_features, dtype=np.float32)
            
            # **确保 feature_vector 是 torch.Tensor**
            feature_vector = torch.tensor(feature_vector, dtype=torch.float32, device=self.device)
            batch_features.append(feature_vector)

        return torch.stack(batch_features)  # 转换为 [batch_size, embedding_dim] 形式



class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                         use_time_proj=True):
        embeddings = memory[source_nodes, :]
        # Log embeddings
        self.log_embedding(source_nodes, timestamps, embeddings)
        return embeddings


class TimeEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
        super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                           neighbor_finder, time_encoder, n_layers,
                                           n_node_features, n_edge_features, n_time_features,
                                           embedding_dimension, device, dropout)

        class NormalLinear(nn.Linear):
            # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.n_node_features)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                         use_time_proj=True):
        source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))
        # Log embeddings
        self.log_embedding(source_nodes, timestamps, source_embeddings)
        return source_embeddings


class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                            neighbor_finder, time_encoder, n_layers,
                                            n_node_features, n_edge_features, n_time_features,
                                            embedding_dimension, device, dropout)

        self.use_memory = use_memory
        self.device = device


    def compute_embedding(self, memory, source_nodes, timestamps, node_features, 
                        n_layers, n_neighbors=20, time_diffs=None, use_time_proj=True):
        """递归实现时间图注意力层的计算"""
        assert (n_layers >= 0)

        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

        # 获取当前时间步的节点特征
        source_node_features = self.get_node_features_at_time(source_nodes_torch, timestamps_torch)

        if self.use_memory:
            source_node_features = memory[source_nodes, :] + source_node_features

        if n_layers == 0:
            embeddings = source_node_features
            self.log_embedding(source_nodes, timestamps, embeddings)
            return embeddings
        else:
            # ✅ 递归调用时，正确提取邻居的 `node_features`
            neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
                source_nodes,
                timestamps,
                n_neighbors=n_neighbors
            )

            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
            edge_deltas = timestamps[:, np.newaxis] - edge_times
            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            # ✅ 只提取需要的邻居特征
            neighbor_node_features = self.get_node_features_at_time(neighbors.flatten(), 
                                                                    np.repeat(timestamps, n_neighbors))

            # ✅ 递归调用时，确保 `node_features` 只包含邻居的特征
            neighbor_embeddings = self.compute_embedding(
                memory,
                neighbors.flatten(),
                np.repeat(timestamps, n_neighbors),
                neighbor_node_features,  # ✅ 这里修正
                n_layers=n_layers - 1,
                n_neighbors=n_neighbors
            )

            # 计算邻居特征
            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
            edge_time_embeddings = self.time_encoder(edge_deltas_torch)

            source_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))

            edge_features = self.edge_features[edge_idxs, :]
            mask = neighbors_torch == 0

            source_embedding = self.aggregate(n_layers, source_node_features,
                                            source_nodes_time_embedding,
                                            neighbor_embeddings,
                                            edge_time_embeddings,
                                            edge_features,
                                            mask)

            self.log_embedding(source_nodes, timestamps, source_embedding)
        return source_embedding


    def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        return NotImplemented


class GraphSumEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                               edge_features=edge_features,
                                               memory=memory,
                                               neighbor_finder=neighbor_finder,
                                               time_encoder=time_encoder,
                                               n_layers=n_layers,
                                               n_node_features=n_node_features,
                                               n_edge_features=n_edge_features,
                                               n_time_features=n_time_features,
                                               embedding_dimension=embedding_dimension,
                                               device=device,
                                               n_heads=n_heads,
                                               dropout=dropout,
                                               use_memory=use_memory)
        self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                            n_edge_features, embedding_dimension)
                                            for _ in range(n_layers)])
        self.linear_2 = torch.nn.ModuleList(
            [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                            embedding_dimension) for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                       dim=2)
        neighbor_embeddings_transformed = self.linear_1[n_layer - 1](neighbors_features)
        neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings_transformed, dim=1))

        source_features = torch.cat([source_node_features,
                                     source_nodes_time_embedding.squeeze()], dim=1)
        source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
        source_embedding = self.linear_2[n_layer - 1](source_embedding)

        return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphAttentionEmbedding, self).__init__(node_features=node_features,
                                                     edge_features=edge_features,
                                                     memory=memory,
                                                     neighbor_finder=neighbor_finder,
                                                     time_encoder=time_encoder,
                                                     n_layers=n_layers,
                                                     n_node_features=n_node_features,
                                                     n_edge_features=n_edge_features,
                                                     n_time_features=n_time_features,
                                                     embedding_dimension=embedding_dimension,
                                                     device=device,
                                                     n_heads=n_heads,
                                                     dropout=dropout,
                                                     use_memory=use_memory)

        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=n_node_features,
            n_neighbors_features=n_node_features,
            n_edge_features=n_edge_features,
            time_dim=n_time_features,
            output_dimension=n_node_features,
            n_head=n_heads,
            dropout=dropout)
            for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        attention_model = self.attention_models[n_layer - 1]

        attn_output, attn_output_weights = attention_model(source_node_features,
                                                           source_nodes_time_embedding,
                                                           neighbor_embeddings,
                                                           edge_time_embeddings,
                                                           edge_features,
                                                           mask)

        return attn_output


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                        time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                        embedding_dimension, device,
                        n_heads=2, dropout=0.1, n_neighbors=None,
                        use_memory=True):
    if module_type == "graph_attention":
        return GraphAttentionEmbedding(node_features=node_features,
                                       edge_features=edge_features,
                                       memory=memory,
                                       neighbor_finder=neighbor_finder,
                                       time_encoder=time_encoder,
                                       n_layers=n_layers,
                                       n_node_features=n_node_features,
                                       n_edge_features=n_edge_features,
                                       n_time_features=n_time_features,
                                       embedding_dimension=embedding_dimension,
                                       device=device,
                                       n_heads=n_heads, dropout=dropout, use_memory=use_memory)
    elif module_type == "graph_sum":
        return GraphSumEmbedding(node_features=node_features,
                                 edge_features=edge_features,
                                 memory=memory,
                                 neighbor_finder=neighbor_finder,
                                 time_encoder=time_encoder,
                                 n_layers=n_layers,
                                 n_node_features=n_node_features,
                                 n_edge_features=n_edge_features,
                                 n_time_features=n_time_features,
                                 embedding_dimension=embedding_dimension,
                                 device=device,
                                 n_heads=n_heads, dropout=dropout, use_memory=use_memory)

    elif module_type == "identity":
        return IdentityEmbedding(node_features=node_features,
                                 edge_features=edge_features,
                                 memory=memory,
                                 neighbor_finder=neighbor_finder,
                                 time_encoder=time_encoder,
                                 n_layers=n_layers,
                                 n_node_features=n_node_features,
                                 n_edge_features=n_edge_features,
                                 n_time_features=n_time_features,
                                 embedding_dimension=embedding_dimension,
                                 device=device,
                                 dropout=dropout)
    elif module_type == "time":
        return TimeEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout,
                             n_neighbors=n_neighbors)
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))
