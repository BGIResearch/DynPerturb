import numpy as np
import random
import pandas as pd
import pickle
import ast


class Data:
    def __init__(
        self, sources, destinations, timestamps, edge_idxs, labels, celltypes=None
    ):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.celltypes = celltypes
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


# General data loader function supporting node classification/link prediction, single/multi-label
def get_data(
    dataset_name,
    use_ddp=False,
    randomize_features=False,
    different_new_nodes_between_val_and_test=False,
    use_validation=False,
    label_processing=False,  # If True, treat as multi-label task
):
    """
    General data loader for node classification/link prediction, single/multi-label.
    Returns: (full_data, node_features, edge_features, train_data, val_data, test_data, new_node_val_data, new_node_test_data)
    """
    # Load data
    graph_df = pd.read_csv(f"./data/{dataset_name}.csv")
    edge_features = np.load(f"./data/{dataset_name}.npy")
    with open(f"./data/{dataset_name}_node.pkl", "rb") as f:
        node_features = pickle.load(f)

    # Sort by timestamp and reset index
    sorted_idx = np.argsort(graph_df.ts.values)
    graph_df = graph_df.iloc[sorted_idx].reset_index(drop=True)
    edge_features = edge_features[sorted_idx]

    # Randomize node features if required
    if randomize_features:
        node_features = np.random.rand(*node_features.shape)

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values - 1
    timestamps = graph_df.ts.values

    # Label processing: multi-label or single-label
    if label_processing:
        raw_labels = graph_df.label.values
        parsed_labels = [ast.literal_eval(x) if isinstance(x, str) else x for x in raw_labels]
        labels = np.array(parsed_labels, dtype=np.float32)
    else:
        labels = graph_df.label.values.astype(int)

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    # Split dataset: train, val, test
    num = len(graph_df)
    train_end, val_end = int(0.70 * num), int(0.85 * num)
    train_mask_raw = np.arange(num) < train_end
    val_mask = (np.arange(num) >= train_end) & (np.arange(num) < val_end)
    test_mask = np.arange(num) >= val_end

    # Sample new nodes only from test set
    test_sources = sources[test_mask]
    test_destinations = destinations[test_mask]
    test_node_set = set(test_sources) | set(test_destinations)
    all_nodes = set(sources) | set(destinations)
    sample_size = min(int(0.1 * len(all_nodes)), len(test_node_set))
    new_nodes = (set(random.sample(list(test_node_set), sample_size)) if sample_size > 0 else set())

    # Only for DDP: remove edges in train set that involve new nodes
    if use_ddp:
        u_mask = graph_df.u.map(lambda x: x in new_nodes).values
        i_mask = graph_df.i.map(lambda x: x in new_nodes).values
        observed_edges_mask = np.logical_and(~u_mask, ~i_mask)
        train_mask = np.logical_and(train_mask_raw, observed_edges_mask)
    else:
        train_mask = train_mask_raw

    # Build Data objects for each split
    train_data = Data(sources[train_mask],destinations[train_mask],timestamps[train_mask],edge_idxs[train_mask],labels[train_mask])
    val_data = Data(sources[val_mask],destinations[val_mask],timestamps[val_mask],edge_idxs[val_mask],labels[val_mask])
    test_data = Data(sources[test_mask],destinations[test_mask],timestamps[test_mask],edge_idxs[test_mask],labels[test_mask])

    # Build new node val/test splits
    edge_contains_new_node_mask = np.array([(u in new_nodes or v in new_nodes) for u, v in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    new_node_val_data = Data(sources[new_node_val_mask],destinations[new_node_val_mask],timestamps[new_node_val_mask],edge_idxs[new_node_val_mask],labels[new_node_val_mask],)
    new_node_test_data = Data(sources[new_node_test_mask],destinations[new_node_test_mask],timestamps[new_node_test_mask],edge_idxs[new_node_test_mask],labels[new_node_test_mask],)

    # Print dataset statistics
    print(f"[Stats] Train: {train_data.n_interactions} interactions, {train_data.n_unique_nodes} nodes")
    print(f"[Stats] Val: {val_data.n_interactions} interactions, {val_data.n_unique_nodes} nodes")
    print(f"[Stats] Test: {test_data.n_interactions} interactions, {test_data.n_unique_nodes} nodes")
    print(f"[Stats] New node val: {new_node_val_data.n_interactions} interactions")
    print(f"[Stats] New node test: {new_node_test_data.n_interactions} interactions")

    return (full_data,node_features,edge_features,train_data,val_data,test_data,new_node_val_data,new_node_test_data,)


# Compute mean and std of time differences for temporal encoding
def compute_time_statistics(sources, destinations, timestamps):
    """
    Compute mean and std of time differences for temporal encoding.
    Returns: (mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst)
    """
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return (mean_time_shift_src,std_time_shift_src,mean_time_shift_dst,std_time_shift_dst,)
