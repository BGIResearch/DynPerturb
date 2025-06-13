import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module_1223 import get_embedding_module
from model.time_encoding import TimeEncode


class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features_dict, edge_features, device, n_layers=2,
                 n_heads=2, dropout=0.1, use_memory=False,
                 memory_update_at_start=True, message_dimension=100,
                 memory_dimension=500, embedding_module_type="graph_attention",
                 message_function="mlp",
                 mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
                 std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
                 memory_updater_type="gru",
                 use_destination_embedding_in_message=False,
                 use_source_embedding_in_message=False,
                 dyrep=False, node_id_map=None):
        super(TGN, self).__init__()

        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.node_features_dict = node_features_dict  # 动态特征字典
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

        self.n_edge_features = self.edge_raw_features.shape[1]
        self.embedding_dimension = 150  # 假设节点特征维度是150
        self.n_neighbors = n_neighbors
        self.embedding_module_type = embedding_module_type
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message
        self.dyrep = dyrep

        self.use_memory = use_memory
        self.time_encoder = TimeEncode(dimension=self.embedding_dimension)
        self.memory = None

        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst
        self.node_id_map = node_id_map  # 用于节点ID映射


        if self.use_memory:
            self.memory_dimension = memory_dimension
            self.memory_update_at_start = memory_update_at_start
            raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                                    self.time_encoder.dimension
            message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
            self.memory = Memory(n_nodes=max(node_features_dict.keys())[0] + 1,  # 动态节点数
                                 memory_dimension=self.memory_dimension,
                                 input_dimension=message_dimension,
                                 message_dimension=message_dimension,
                                 device=device)
            self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                             device=device)
            self.message_function = get_message_function(module_type=message_function,
                                                         raw_message_dimension=raw_message_dimension,
                                                         message_dimension=message_dimension)
            self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                     memory=self.memory,
                                                     message_dimension=message_dimension,
                                                     memory_dimension=self.memory_dimension,
                                                     device=device)

        self.embedding_module_type = embedding_module_type

        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                     node_features=self.node_features_dict,  # 动态加载特征
                                                     edge_features=self.edge_raw_features,
                                                     memory=self.memory,
                                                     neighbor_finder=self.neighbor_finder,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     n_node_features=self.embedding_dimension,
                                                     n_edge_features=self.n_edge_features,
                                                     n_time_features=self.embedding_dimension,
                                                     embedding_dimension=self.embedding_dimension,
                                                     device=self.device,
                                                     n_heads=n_heads, dropout=dropout,
                                                     use_memory=use_memory,
                                                     n_neighbors=self.n_neighbors)

        # MLP to compute probability on an edge given two node embeddings
        self.affinity_score = MergeLayer(self.embedding_dimension, self.embedding_dimension,
                                         self.embedding_dimension,
                                         1) 
        
  def get_node_features(self, nodes, timestamps, celltypes):
      features = []
      for node_id, timestamp, celltype in zip(nodes, timestamps, celltypes):
          # 使用 node_id_map 来将节点 ID 映射为连续索引
          #node_continuous_idx = self.node_id_map[node_id]

          # 获取节点特征
          #feature = self.node_features_dict.get((node_continuous_idx, timestamp, celltype))
          feature = self.node_features_dict.get((node_id, timestamp, celltype))

          if feature is not None:
              # 确保特征是张量类型
              if isinstance(feature, np.ndarray):  # 如果是 numpy 数组，转换为 tensor
                  feature = torch.tensor(feature)
                  
              if feature.ndimension() == 1:
                  feature = feature.unsqueeze(0)  # 确保 feature 的维度是 (1, 150)
              features.append(feature)
          else:
              # 如果没有特征，返回零向量
              features.append(torch.zeros(1, 150))  # 默认返回零向量

      # 将所有特征堆叠成一个 tensor
      return torch.stack(features)


  
  # def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
  #                               edge_idxs, n_neighbors=20, source_features=None, destination_features=None):
  #   """
  #   计算源节点、目标节点以及负采样节点的时间嵌入。
  #   """
  #   # 使用 node_id_map 将节点 ID 转换为连续索引
  #   # source_nodes = [self.node_id_map[node_id] for node_id in source_nodes]
  #   # destination_nodes = [self.node_id_map[node_id] for node_id in destination_nodes]
  #   # negative_nodes = [self.node_id_map[node_id] for node_id in negative_nodes]

  #   # 获取源节点、目标节点、负采样节点的特征
  #   source_features = self.get_node_features(source_nodes, edge_times, edge_times)
  #   destination_features = self.get_node_features(destination_nodes, edge_times, edge_times)
  #   negative_features = self.get_node_features(negative_nodes, edge_times, edge_times)

  #   memory = None
  #   time_diffs = None
  #   if self.use_memory:
  #       # 计算动态节点数，避免使用硬编码的 self.n_nodes
  #       n_nodes = len(set(source_nodes).union(destination_nodes).union(negative_nodes))  # 计算唯一节点数
  #       memory, last_update = self.get_updated_memory(list(range(n_nodes)), self.memory.messages)

  #       # 计算源节点和目标节点的时间差
  #       source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[source_nodes].long()
  #       source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
  #       destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[destination_nodes].long()
  #       destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
  #       time_diffs = torch.cat([source_time_diffs, destination_time_diffs], dim=0)

  #   # 将节点转换为张量
  #   source_nodes_torch = torch.tensor(source_nodes).to(self.device)
  #   destination_nodes_torch = torch.tensor(destination_nodes).to(self.device)
  #   negative_nodes_torch = torch.tensor(negative_nodes).to(self.device)

  #   # 计算时间嵌入
  #   node_embedding = self.embedding_module.compute_embedding(
  #       memory=memory,
  #       source_nodes=source_nodes,
  #       timestamps=edge_times,
  #       n_layers=self.n_layers,
  #       n_neighbors=n_neighbors,
  #       time_diffs=time_diffs
  #   )
    

  #   return source_features, destination_features, negative_features

  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,celltypes,
                                edge_idxs, n_neighbors=20, source_features=None, destination_features=None):
    
    # 获取源节点、目标节点、负采样节点的特征
    source_features = self.get_node_features(source_nodes, edge_times,celltypes)
    destination_features = self.get_node_features(destination_nodes, edge_times,celltypes)
    negative_features = self.get_node_features(negative_nodes, edge_times ,celltypes)

    # 内存和时间差初始化
    memory = None
    time_diffs = None

    if self.use_memory:
        # 动态计算节点数，避免使用硬编码的 self.n_nodes
        n_nodes = len(set(source_nodes).union(destination_nodes).union(negative_nodes))  # 计算唯一节点数
        memory, last_update = self.get_updated_memory(list(range(n_nodes)), self.memory.messages)

        # 计算源节点和目标节点的时间差
        source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[source_nodes].long()
        source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
        destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[destination_nodes].long()
        destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

        # 合并时间差
        time_diffs = torch.cat([source_time_diffs, destination_time_diffs], dim=0)

    # 将节点 ID 转换为张量
    source_nodes_torch = torch.tensor(source_nodes).to(self.device)
    destination_nodes_torch = torch.tensor(destination_nodes).to(self.device)
    negative_nodes_torch = torch.tensor(negative_nodes).to(self.device)

    # 计算时间嵌入
    node_embedding = self.embedding_module.compute_embedding(
        memory=memory,
        source_nodes=source_nodes_torch,
        timestamps=edge_times,
        celltypes=celltypes,
        n_layers=self.n_layers,
        n_neighbors=n_neighbors,
        time_diffs=time_diffs
    )

    # 将节点嵌入拆分为源节点、目标节点和负节点的嵌入
    n_samples = len(source_nodes)
    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    # 更新内存（如果启用内存）
    if self.use_memory:
        if self.memory_update_at_start:
            # 持久化源节点和目标节点的内存更新
            self.update_memory([source_nodes, destination_nodes], self.memory.messages)

            # 检查内存更新是否正确
            assert torch.allclose(memory[source_nodes], self.memory.get_memory(source_nodes), atol=1e-3), \
                "Memory update issue for source nodes"

            assert torch.allclose(memory[destination_nodes], self.memory.get_memory(destination_nodes), atol=1e-3), \
                "Memory update issue for destination nodes"

            # 清除消息
            self.memory.clear_messages([source_nodes, destination_nodes])

        # 获取源节点和目标节点的原始消息
        unique_sources, source_id_to_messages = self.get_raw_messages(
            source_nodes, source_node_embedding, destination_nodes, destination_node_embedding, edge_times, edge_idxs
        )
        unique_destinations, destination_id_to_messages = self.get_raw_messages(
            destination_nodes, destination_node_embedding, source_nodes, source_node_embedding, edge_times, edge_idxs
        )

        # 存储或更新内存中的原始消息
        if self.memory_update_at_start:
            self.memory.store_raw_messages(unique_sources, source_id_to_messages)
            self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
        else:
            self.update_memory(unique_sources, source_id_to_messages)
            self.update_memory(unique_destinations, destination_id_to_messages)

        # 如果使用动态表示（dyrep），直接从内存获取嵌入
        if self.dyrep:
            source_node_embedding = memory[source_nodes]
            destination_node_embedding = memory[destination_nodes]
            negative_node_embedding = memory[negative_nodes]

    return source_node_embedding, destination_node_embedding, negative_node_embedding

  
  def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,celltypes,
                               edge_idxs, n_neighbors=20):
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    """
    # source_nodes = [self.node_id_map[node_id] for node_id in source_nodes]
    # destination_nodes = [self.node_id_map[node_id] for node_id in destination_nodes]
    # negative_nodes = [self.node_id_map[node_id] for node_id in negative_nodes]

    source_features, destination_features, negative_features = self.compute_temporal_embeddings(
        source_nodes, destination_nodes, negative_nodes, edge_times,celltypes, edge_idxs, n_neighbors)

    score = self.affinity_score(torch.cat([source_features, source_features], dim=0),
                                torch.cat([destination_features,
                                           negative_features])).squeeze(dim=0)
    pos_score = score[:len(source_nodes)]
    neg_score = score[len(source_nodes):]

    return pos_score.sigmoid(), neg_score.sigmoid()



  def update_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)
  

  
  def get_updated_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update


  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
