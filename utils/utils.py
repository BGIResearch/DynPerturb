import numpy as np
import torch
import torch.nn.functional as F

# MergeLayer: Merges two input tensors and applies two linear layers with ReLU activation
class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)

# MLP: Multi-layer perceptron for classification
class MLP(torch.nn.Module):
    def __init__(self, device, dim, num_classes=None, drop=0.3):
        super().__init__()
        self.device = device
        print("num:", num_classes)
        print(f"Using device: {device}")
        self.fc_1 = torch.nn.Linear(dim, 80)  # Input to hidden layer
        self.fc_2 = torch.nn.Linear(80, 10)   # Hidden to second layer
        self.fc_3 = torch.nn.Linear(10, num_classes)  # Output layer
        self.act = torch.nn.ReLU()            # Activation function
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)  # Dropout layer

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x)

# Early stopping monitor for single or multi-task training
class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.best_model_state = None
        self.last_best = None
        self.last_best_edge = None  # For best edge prediction result
        self.last_best_node = None  # For best node classification result
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check_raw(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round

    def early_stop_check(self, curr_val_edge, curr_val_node):
        """
        Early stopping for two tasks (edge prediction and node classification).
        Only stop if both tasks meet the stopping condition.
        """
        if not self.higher_better:
            curr_val_edge *= -1
            curr_val_node *= -1
        if self.last_best_edge is None or self.last_best_node is None:
            self.last_best_edge = curr_val_edge
            self.last_best_node = curr_val_node
        elif (curr_val_edge - self.last_best_edge) / np.abs(self.last_best_edge) > self.tolerance and (curr_val_node - self.last_best_node) / np.abs(self.last_best_node) > self.tolerance:
            self.last_best_edge = curr_val_edge
            self.last_best_node = curr_val_node
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round

# Early stopping monitor for DDP training
class EarlyStopMonitor_ddp(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.best_model_state = None
        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round


# Random edge sampler for negative sampling
default_seed = None
class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:
            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)

# Build neighbor finder for temporal graph
def get_neighbor_finder(data, uniform, max_node_idx=None):
    max_node_idx = (max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx)
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations, data.edge_idxs, data.timestamps):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))
    return NeighborFinder(adj_list, uniform=uniform)

# Temporal neighbor finder for dynamic graphs
class NeighborFinder:
    def __init__(self, adj_list, uniform=False, seed=None):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []
        for neighbors in adj_list:
            # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
            # Sort by timestamp
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))
        self.uniform = uniform
        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def find_before(self, src_idx, cut_time):
        """
        Extract all interactions before cut_time for user src_idx in the graph.
        Returns 3 lists: neighbors, edge_idxs, timestamps (sorted by time).
        """
        if len(self.node_to_edge_timestamps[src_idx]) == 0:
            print(f"Warning: Node {src_idx} has no timestamps.")
            return [], [], []
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
        return (self.node_to_neighbors[src_idx][:i],self.node_to_edge_idxs[src_idx][:i],self.node_to_edge_timestamps[src_idx][:i])

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        """
        For each source node and timestamp, sample a temporal neighborhood of up to n_neighbors.
        Returns arrays of neighbor ids, edge times, and edge indices for each source node.
        """
        assert len(source_nodes) == len(timestamps)
        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors), dtype=np.int32)
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors), dtype=np.float32)
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors), dtype=np.int32)
        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node, timestamp)
            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)
                    neighbors[i, :] = source_neighbors[sampled_idx]
                    edge_times[i, :] = source_edge_times[sampled_idx]
                    edge_idxs[i, :] = source_edge_idxs[sampled_idx]
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                else:
                    source_edge_times = source_edge_times[-n_neighbors:]
                    source_neighbors = source_neighbors[-n_neighbors:]
                    source_edge_idxs = source_edge_idxs[-n_neighbors:]
                    assert len(source_neighbors) <= n_neighbors
                    assert len(source_edge_times) <= n_neighbors
                    assert len(source_edge_idxs) <= n_neighbors
                    neighbors[i, n_neighbors - len(source_neighbors) :] = (source_neighbors)
                    edge_times[i, n_neighbors - len(source_edge_times) :] = (source_edge_times)
                    edge_idxs[i, n_neighbors - len(source_edge_idxs) :] = (source_edge_idxs)
        return neighbors, edge_idxs, edge_times
