import numpy as np
import torch
from collections import defaultdict
from utils.utils import MergeLayer, MLP
from modules.MemoryModule import Memory
from modules.MessageOps import get_message_aggregator, get_message_function
from modules.MemoryModule import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode


# DynPerturbModel: Main Model Class
class DynPerturbModel(torch.nn.Module):
    """
    Temporal Graph Neural Network Model for link prediction and node classification.
    """

    def __init__(self, num_nodes, neighbor_finder, node_features, edge_features,
        device, n_layers=2, n_heads=2, dropout=0.1, use_memory=False,
        num_classes=None, memory_update_at_start=True, message_dimension=100,
        memory_dimension=500, embedding_module_type="graph_attention",
        message_function="mlp", mean_time_shift_src=0, std_time_shift_src=1,
        mean_time_shift_dst=0, std_time_shift_dst=1, n_neighbors=None,
        aggregator_type="last", memory_updater_type="gru",
        use_destination_embedding_in_message=False, use_source_embedding_in_message=False,
        dyrep=False, mode="link_prediction"):
        super(DynPerturbModel, self).__init__()

        # Model hyperparameters
        self.n_layers = n_layers  # Number of GNN layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.node_features = node_features
        self.node_raw_features = node_features
        self.mode = mode  # Task mode

        if self.mode not in ["link_prediction", "node_classification"]:
            raise ValueError("Mode must be either 'link_prediction' or 'node_classification'")

        # Task-specific parameters
        if self.mode == "link_prediction":
            self.num_classes = None
        elif self.mode == "node_classification":
            self.num_classes = num_classes  # Number of classes for node classification

        # Node and feature dimensions
        self.n_nodes = num_nodes  # Number of nodes in the graph
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)
        self.n_node_features = memory_dimension
        self.n_edge_features = self.edge_raw_features.shape[1]
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors
        self.embedding_module_type = embedding_module_type 
        self.use_destination_embedding_in_message = use_destination_embedding_in_message  # Use destination embedding in message
        self.use_source_embedding_in_message = use_source_embedding_in_message  # Use source embedding in message
        self.dyrep = dyrep

        # Time and memory modules
        self.use_memory = use_memory
        self.time_encoder = TimeEncode(dimension=self.n_node_features)  # Time encoding module
        self.memory = None
        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst

        # Memory and message modules initialization
        if self.use_memory:
            self.memory_dimension = memory_dimension
            self.memory_update_at_start = memory_update_at_start

            # Calculate message dimensions
            raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + self.time_encoder.dimension
            message_dimension = message_dimension if message_function != "identity" else raw_message_dimension

            # Module initializations
            self.memory = Memory(n_nodes=self.n_nodes, memory_dimension=self.memory_dimension, input_dimension=message_dimension, message_dimension=message_dimension, device=device, mode=self.mode)
            self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type, device=device)
            self.message_function = get_message_function(module_type=message_function, raw_message_dimension=raw_message_dimension, message_dimension=message_dimension)
            self.memory_updater = get_memory_updater(module_type=memory_updater_type, memory=self.memory, message_dimension=message_dimension, memory_dimension=self.memory_dimension, device=device, mode=self.mode)
        
        # Initialize embedding module for temporal graph learning
        self.embedding_module_type = embedding_module_type
        self.embedding_module = get_embedding_module(module_type=embedding_module_type,node_features=self.node_raw_features,
            edge_features=self.edge_raw_features,memory=self.memory,neighbor_finder=self.neighbor_finder,
            time_encoder=self.time_encoder,n_layers=self.n_layers,n_node_features=self.n_node_features,
            n_edge_features=self.n_edge_features,n_time_features=self.n_node_features,embedding_dimension=self.embedding_dimension,
            device=self.device,n_heads=n_heads,dropout=dropout,use_memory=use_memory,n_neighbors=self.n_neighbors)

        # Decoder for node classification
        if self.mode == "node_classification":
            # Use MLP as node classification decoder
            self.node_classification_decoder = MLP(device, memory_dimension, num_classes, dropout)

        # MergeLayer for link prediction affinity score
        self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features, self.n_node_features, 1)

    # Compute temporal embeddings for nodes
    def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors=20):
        """
        Compute temporal embeddings for sources, destinations, and negatives.
        :param source_nodes: [batch_size] source ids
        :param destination_nodes: [batch_size] destination ids
        :param negative_nodes: [batch_size] negative sampled destination ids
        :param edge_times: [batch_size] timestamps of interactions
        :param edge_idxs: [batch_size] indices of interactions
        :param n_neighbors: number of temporal neighbors to consider in each convolutional layer
        :return: Temporal embeddings for sources, destinations, and negatives
        """
        n_samples = len(source_nodes)

        # Prepare node and time arrays for embedding
        if self.mode == "link_prediction":
            nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
            positives = np.concatenate([source_nodes, destination_nodes])
        else:
            nodes = np.concatenate([(source_nodes.cpu().numpy() if isinstance(source_nodes, torch.Tensor) else source_nodes), (destination_nodes.cpu().numpy() if isinstance(destination_nodes, torch.Tensor) else destination_nodes), (negative_nodes.cpu().numpy() if isinstance(negative_nodes, torch.Tensor) else negative_nodes)])
            positives = np.concatenate([(source_nodes.cpu().numpy() if isinstance(source_nodes, torch.Tensor) else source_nodes), (destination_nodes.cpu().numpy() if isinstance(destination_nodes, torch.Tensor) else destination_nodes)])
            edge_times = edge_times.cpu().numpy() if isinstance(edge_times, torch.Tensor) else edge_times

        # Prepare timestamps and node features for embedding
        timestamps = np.concatenate([edge_times, edge_times, edge_times])
        node_features = self.embedding_module.get_node_features_at_time(nodes, timestamps)
        memory = None
        
        # Placeholder for time differences, will be set if memory is used
        time_diffs = None

        # Memory handling
        if self.use_memory:
            if self.memory_update_at_start:
                memory, last_update = self.get_updated_memory(list(range(self.n_nodes)), self.memory.messages)
            else:
                if self.mode == "link_prediction":
                    memory = self.memory.get_memory(list(range(self.n_nodes)))
                else:
                    memory = self.memory.get_memory(list(range(self.n_nodes))).detach().clone().requires_grad_(True)
                last_update = self.memory.last_update
                
            # Calculate time differences for each node type
            source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[source_nodes].long()
            source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src

            destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[destination_nodes].long()
            destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

            negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[negative_nodes].long()
            negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

            # Concatenate all time differences for embedding
            time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs], dim=0)
        
        node_embedding = self.embedding_module.compute_embedding(memory=memory, source_nodes=nodes, timestamps=timestamps, node_features=node_features, n_layers=self.n_layers, n_neighbors=n_neighbors, time_diffs=time_diffs)
        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples : 2 * n_samples]
        negative_node_embedding = node_embedding[2 * n_samples :]

        # Memory update if enabled
        if self.use_memory:
            if self.memory_update_at_start:
                # Persist the updates to the memory only for sources and destinations
                self.update_memory(positives, self.memory.messages)
                assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-3), "Memory update mismatch after update"
                self.memory.clear_messages(positives)
                
            # Collect and update messages for sources and destinations
            unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes, source_node_embedding, destination_nodes, destination_node_embedding, edge_times, edge_idxs)
            unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes, destination_node_embedding, source_nodes, source_node_embedding, edge_times, edge_idxs)
            
            if self.memory_update_at_start: 
                self.memory.store_raw_messages(unique_sources, source_id_to_messages)
                self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
            else:
                self.update_memory(unique_sources, source_id_to_messages)
                self.update_memory(unique_destinations, destination_id_to_messages)

            # DyRep mode: use memory as embedding
            if self.dyrep:
                source_node_embedding = memory[source_nodes]
                destination_node_embedding = memory[destination_nodes]
                negative_node_embedding = memory[negative_nodes]

        return source_node_embedding, destination_node_embedding, negative_node_embedding

    # Compute edge probabilities for link prediction
    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors=20):
        """
        Compute probabilities for edges between sources and destinations and negatives.
        :param source_nodes: [batch_size] source ids
        :param destination_nodes: [batch_size] destination ids
        :param negative_nodes: [batch_size] negative sampled destination ids
        :param edge_times: [batch_size] timestamps of interactions
        :param edge_idxs: [batch_size] indices of interactions
        :param n_neighbors: number of temporal neighbors to consider in each convolutional layer
        :return: Probabilities for both the positive and negative edges
        """
        n_samples = len(source_nodes)
        source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)
        
        # Compute link prediction scores for positive and negative samples
        score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0), torch.cat([destination_node_embedding, negative_node_embedding])).squeeze(dim=0)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]
        
        return pos_score.sigmoid(), neg_score.sigmoid()

    # Update memory for nodes
    def update_memory(self, nodes, messages):
        """
        Aggregate messages for the same nodes and update the memory.
        :param nodes: List of node ids
        :param messages: Dictionary mapping node id to list of (message, timestamp)
        """
        # Aggregate messages and update memory for nodes
        unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes, messages)
        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)
            
        self.memory_updater.update_memory(unique_nodes, unique_messages, timestamps=unique_timestamps)

    # Get updated memory and last update for nodes
    def get_updated_memory(self, nodes, messages):
        """
        Aggregate messages for the same nodes and return updated memory and last update.
        :param nodes: List of node ids
        :param messages: Dictionary mapping node id to list of (message, timestamp)
        :return: Updated memory and last update
        """
        # Aggregate messages for nodes
        unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(nodes, messages)
        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        # Get updated memory and last update
        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes, unique_messages, timestamps=unique_timestamps)
        
        return updated_memory, updated_last_update

    # Construct raw messages for each source node
    def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes, destination_node_embedding, edge_times, edge_idxs):
        """
        Construct raw messages for each source node, including memory, destination memory, edge features, and time encoding.
        :param source_nodes: List of source node ids
        :param source_node_embedding: Embeddings for source nodes
        :param destination_nodes: List of destination node ids
        :param destination_node_embedding: Embeddings for destination nodes
        :param edge_times: List or array of edge timestamps
        :param edge_idxs: List or array of edge indices
        :return: Unique source node ids and a dictionary mapping node id to list of (message, timestamp)
        """
        # Prepare edge times and features
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
        edge_features = self.edge_raw_features[edge_idxs]

        # Convert source nodes to numpy if needed
        if isinstance(source_nodes, torch.Tensor):
            source_nodes_np = source_nodes.cpu().numpy()
        else:
            source_nodes_np = source_nodes

        # Get unique source nodes
        unique_sources = np.unique(source_nodes_np)

        # Gather memory and embedding for message construction
        source_memory = self.memory.get_memory(source_nodes) if not self.use_source_embedding_in_message else source_node_embedding
        destination_memory = self.memory.get_memory(destination_nodes) if not self.use_destination_embedding_in_message else destination_node_embedding

        # Compute time delta and encoding for each source node
        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(source_nodes), -1)

        # Concatenate all message components
        source_message = torch.cat([source_memory, destination_memory, edge_features, source_time_delta_encoding], dim=1)

        # Build message dictionary for each source node
        messages = defaultdict(list)
        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i], edge_times[i]))

        return unique_sources, messages

    # Set a new neighbor finder for the model
    def set_neighbor_finder(self, neighbor_finder):
        """
        Set a new neighbor finder for the model and its embedding module.
        :param neighbor_finder: New neighbor finder object
        """
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder

    # Forward pass for node classification mode
    def forward(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors=20):
        """
        Forward pass for node classification mode. Computes embeddings and returns link prediction and classification outputs.
        :param source_nodes: List of source node ids
        :param destination_nodes: List of destination node ids
        :param negative_nodes: List of negative sampled destination node ids
        :param edge_times: List or array of edge timestamps
        :param edge_idxs: List or array of edge indices
        :param n_neighbors: Number of neighbors to use
        :return: Positive/negative link prediction scores and node classification logits
        """
        # Only valid for node classification mode
        if self.mode != "node_classification":
            raise ValueError("Mode must be 'node_classification'")

        # Compute temporal embeddings for all nodes
        src_emb, dst_emb, neg_emb = self.compute_temporal_embeddings(source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

        n_samples = len(source_nodes)

        # Compute link prediction scores
        score = self.affinity_score(torch.cat([src_emb, src_emb], dim=0),torch.cat([dst_emb, neg_emb], dim=0)).squeeze(dim=0)
        pos_score = score[:n_samples].sigmoid()
        neg_score = score[n_samples:].sigmoid()

        # Compute node classification logits
        logits = self.node_classification_decoder(src_emb)

        return pos_score, neg_score, logits
