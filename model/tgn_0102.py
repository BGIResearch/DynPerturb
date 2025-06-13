import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode

def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                edge_idxs, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbors to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = None
    time_diffs = None
    if self.use_memory:
        if self.memory_update_at_start:
            # Update memory for all nodes with messages stored in previous batches
            memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                          self.memory.messages)
        else:
            memory = self.memory.get_memory(list(range(self.n_nodes)))
            last_update = self.memory.last_update

        ### Compute differences between the time the memory of a node was last updated,
        ### and the time for which we want to compute the embedding of a node
        source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
            source_nodes].long()
        source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
        destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
            destination_nodes].long()
        destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
        negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
            negative_nodes].long()
        negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

        time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                               dim=0)

    # 提取节点在给定时间戳的特征
    source_node_features = self.get_node_features_at_time(source_nodes, edge_times)
    destination_node_features = self.get_node_features_at_time(destination_nodes, edge_times)
    negative_node_features = self.get_node_features_at_time(negative_nodes, edge_times)

    # 计算嵌入时，使用提取的节点特征
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)

    # 将提取的节点特征直接作为节点嵌入
    source_node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                   source_nodes=source_nodes,
                                                                   timestamps=edge_times,
                                                                   n_layers=self.n_layers,
                                                                   n_neighbors=n_neighbors)
    destination_node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                        source_nodes=destination_nodes,
                                                                        timestamps=edge_times,
                                                                        n_layers=self.n_layers,
                                                                        n_neighbors=n_neighbors)
    negative_node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                     source_nodes=negative_nodes,
                                                                     timestamps=edge_times,
                                                                     n_layers=self.n_layers,
                                                                     n_neighbors=n_neighbors)

    if self.use_memory:
        if self.memory_update_at_start:
            # Persist the updates to the memory only for sources and destinations (since now we have
            # new messages for them)
            self.update_memory(positives, self.memory.messages)

            assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-3), \
                "Something wrong in how the memory was updated"

            # Remove messages for the positives since we have already updated the memory using them
            self.memory.clear_messages(positives)

        unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
        unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
        if self.memory_update_at_start:
            self.memory.store_raw_messages(unique_sources, source_id_to_messages)
            self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
        else:
            self.update_memory(unique_sources, source_id_to_messages)
            self.update_memory(unique_destinations, destination_id_to_messages)

        if self.dyrep:
            source_node_embedding = memory[source_nodes]
            destination_node_embedding = memory[destination_nodes]
            negative_node_embedding = memory[negative_nodes]

    return source_node_embedding, destination_node_embedding, negative_node_embedding
