import torch
import logging
import numpy as np
from collections import defaultdict
from modules.MemoryModule import Memory
from modules.MessageOps import get_message_aggregator
from modules.MessageOps import get_message_function
from modules.MemoryModule import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode
from utils.utils import MLP
from utils.utils import MergeLayer


class NetModel(torch.nn.Module):
    def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
                n_heads=2, dropout=0.1, use_memory=False, num_classes=None,
                memory_update_at_start=True, message_dimension=100,
                memory_dimension=500, embedding_module_type="graph_attention",
                message_function="mlp",
                mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
                std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
                memory_updater_type="gru",
                use_destination_embedding_in_message=False,
                use_source_embedding_in_message=False,
                dyrep=False):
        super(NetModel, self).__init__()

        self.num_classes = num_classes  # 存储 num_classes

        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.node_features = node_features

        self.node_raw_features = node_features
        # self.node_raw_features = {}
        # all_nodes_set = set()
        # for (node_id, timestamp), feat in node_features.items():
        #     if timestamp not in self.node_raw_features:
        #         self.node_raw_features[timestamp] = {}
        #     self.node_raw_features[timestamp][node_id] = torch.from_numpy(feat.astype(np.float32)).to(device)
        #     all_nodes_set.add(node_id)

        # self.n_nodes = len(all_nodes_set)
        # self.n_nodes = len(node_features)
        self.n_nodes = 88863

        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

        #self.n_node_features = self.node_raw_features.shape[1]
        # first_timestamp = list(self.node_raw_features.keys())[0]
        # first_node_id = list(self.node_raw_features[first_timestamp].keys())[0]
        # self.n_node_features = self.node_raw_features[first_timestamp][first_node_id].shape[0]
        # self.n_node_features=1000
        self.n_node_features = memory_dimension
        ##
        #self.n_nodes = self.node_raw_features.shape[0]
        self.n_edge_features = self.edge_raw_features.shape[1]
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors
        self.embedding_module_type = embedding_module_type
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message
        self.dyrep = dyrep

        self.use_memory = use_memory
        self.time_encoder = TimeEncode(dimension=self.n_node_features)
        self.memory = None

        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst

        if self.use_memory:
            self.memory_dimension = memory_dimension
            self.memory_update_at_start = memory_update_at_start
            raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                                    self.time_encoder.dimension
            message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
            self.memory = Memory(n_nodes=self.n_nodes,
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
                                                    node_features=self.node_raw_features,
                                                    edge_features=self.edge_raw_features,
                                                    memory=self.memory,
                                                    neighbor_finder=self.neighbor_finder,
                                                    time_encoder=self.time_encoder,
                                                    n_layers=self.n_layers,
                                                    n_node_features=self.n_node_features,
                                                    n_edge_features=self.n_edge_features,
                                                    n_time_features=self.n_node_features,
                                                    embedding_dimension=self.embedding_dimension,
                                                    device=self.device,
                                                    n_heads=n_heads, dropout=dropout,
                                                    use_memory=use_memory,
                                                    n_neighbors=self.n_neighbors)

        # # MLP to compute probability on an edge given two node embeddings
        # self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
        #                                  self.n_node_features,
        #                                  1)
        # 如果您的解码器或嵌入层没有适应多分类任务，您需要调整最后的输出层
        # self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features, 
        #                                  self.n_node_features, 1)
        # Decoder for node classification
        self.node_classification_decoder = MLP(memory_dimension, num_classes, dropout)

        # MergeLayer for link prediction
        self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                        self.n_node_features, 1)


    def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                    edge_idxs, n_neighbors=20):
        """
        Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

        source_nodes [batch_size]: source ids.
        :param destination_nodes [batch_size]: destination ids
        :param negative_nodes [batch_size]: ids of negative sampled destination
        :param edge_times [batch_size]: timestamp of interaction
        :param edge_idxs [batch_size]: index of interaction
        :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        layer
        :return: Temporal embeddings for sources, destinations and negatives
        """

        n_samples = len(source_nodes)
        #nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
        nodes = np.concatenate([
            source_nodes.cpu().numpy() if isinstance(source_nodes, torch.Tensor) else source_nodes,
            destination_nodes.cpu().numpy() if isinstance(destination_nodes, torch.Tensor) else destination_nodes,
            negative_nodes.cpu().numpy() if isinstance(negative_nodes, torch.Tensor) else negative_nodes
        ])
        positives = np.concatenate([
            source_nodes.cpu().numpy() if isinstance(source_nodes, torch.Tensor) else source_nodes,
            destination_nodes.cpu().numpy() if isinstance(destination_nodes, torch.Tensor) else destination_nodes
        ])
        edge_times = edge_times.cpu().numpy() if isinstance(edge_times, torch.Tensor) else edge_times
        timestamps = np.concatenate([edge_times, edge_times, edge_times])

        node_features = self.embedding_module.get_node_features_at_time(nodes, timestamps)

        memory = None
        time_diffs = None
        if self.use_memory:
            if self.memory_update_at_start:
                # Update memory for all nodes with messages stored in previous batches
                memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                            self.memory.messages)
            else:
                # memory = self.memory.get_memory(list(range(self.n_nodes)))
                # memory = self.memory.get_memory(list(range(self.n_nodes))).clone()
                memory = self.memory.get_memory(list(range(self.n_nodes))).detach().clone().requires_grad_(True)

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

        # Compute the embeddings using the embedding module
        # node_embedding = self.embedding_module.compute_embedding(memory=memory,
        #                                                          source_nodes=nodes,
        #                                                          timestamps=timestamps,
        #                                                          n_layers=self.n_layers,
        #                                                          n_neighbors=n_neighbors,
        #                                                          time_diffs=time_diffs)

        # modify
        node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                source_nodes=nodes,
                                                                timestamps=timestamps,
                                                                node_features=node_features,
                                                                n_layers=self.n_layers,
                                                                n_neighbors=n_neighbors,
                                                                time_diffs=time_diffs)

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
        negative_node_embedding = node_embedding[2 * n_samples:]

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

    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                    edge_idxs, n_neighbors=20):
        """
        Compute probabilities for edges between sources and destination and between sources and
        negatives by first computing temporal embeddings using the NetModel encoder and then feeding them
        into the MLP decoder.
        :param destination_nodes [batch_size]: destination ids
        :param negative_nodes [batch_size]: ids of negative sampled destination
        :param edge_times [batch_size]: timestamp of interaction
        :param edge_idxs [batch_size]: index of interaction
        :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        layer
        :return: Probabilities for both the positive and negative edges
        """
        n_samples = len(source_nodes)
        source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

        score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                    torch.cat([destination_node_embedding,
                                                negative_node_embedding])).squeeze(dim=0)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

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

        # Check if source_nodes is a tensor, and move it to CPU if it is.
        if isinstance(source_nodes, torch.Tensor):
            source_nodes = source_nodes.cpu().numpy()
        unique_sources = np.unique(source_nodes)

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
        # unique_sources = np.unique(source_nodes)
        # unique_sources = np.unique(source_nodes.cpu().numpy())

        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i], edge_times[i]))

        return unique_sources, messages

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder

    def forward(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                edge_idxs, n_neighbors=20):
        
        # ✅ 只执行一次 embedding 计算
        src_emb, dst_emb, neg_emb = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

        # 🔗 链接预测
        n_samples = len(source_nodes)
        score = self.affinity_score(
            torch.cat([src_emb, src_emb], dim=0),
            torch.cat([dst_emb, neg_emb], dim=0)
        ).squeeze(dim=0)
        pos_score = score[:n_samples].sigmoid()
        neg_score = score[n_samples:].sigmoid()

        # 🔍 节点分类
        logits = self.node_classification_decoder(src_emb)

        return pos_score, neg_score, logits


    # def forward_link_prediction(self, source_nodes, destination_nodes, negative_nodes, edge_times,
    #                             edge_idxs, n_neighbors=20):
    #   src_emb, dst_emb, neg_emb = self.compute_temporal_embeddings(
    #       source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)
    #   n_samples = len(source_nodes)
    #   score = self.affinity_score(
    #       torch.cat([src_emb, src_emb], dim=0),
    #       torch.cat([dst_emb, neg_emb], dim=0)).squeeze(dim=0)
    #   pos_score = score[:n_samples].sigmoid()
    #   neg_score = score[n_samples:].sigmoid()
    #   return pos_score, neg_score

    # def forward_node_classification(self, source_nodes, destination_nodes, negative_nodes, edge_times,
    #                                 edge_idxs, n_neighbors=20):
    #   src_emb, _, _ = self.compute_temporal_embeddings(
    #       source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)
    #   logits = self.node_classification_decoder(src_emb)
    #   return logits

    # def forward(self, source_nodes, destination_nodes, negative_nodes, edge_times,
    #           edge_idxs, n_neighbors=20):
    #   pos_score, neg_score = self.forward_link_prediction(
    #       source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)
    #   logits = self.forward_node_classification(
    #       source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)
    #   return pos_score, neg_score, logits