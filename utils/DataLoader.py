import numpy as np
import random
import pandas as pd
import pickle
import numpy as np
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


# def get_data_node_classification(dataset_name, use_validation=False):
#   ### Load data and train val test split
#   graph_df = pd.read_csv(f'./data/ml_{dataset_name}.csv')
#   edge_features = np.load(f'./data/ml_{dataset_name}.npy')

#   # Read node features
#   with open(f'./data/ml_{dataset_name}_node.pkl', 'rb') as f:
#       node_features = pickle.load(f)

#   val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

#   sources = graph_df.u.values.astype(int)
#   destinations = graph_df.i.values.astype(int)
#   edge_idxs = graph_df.idx.values
#   labels = graph_df.label.values  # For link prediction
#   node_labels = graph_df.node_label.values  # For node classification
#   timestamps = graph_df.ts.values

#   random.seed(2020)

#   train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
#   test_mask = timestamps > test_time
#   val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

#   full_data = Data(sources, destinations, timestamps, edge_idxs, labels, node_labels)

#   train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
#                     edge_idxs[train_mask], labels[train_mask], node_labels[train_mask])

#   val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
#                   edge_idxs[val_mask], labels[val_mask], node_labels[val_mask])

#   test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
#                     edge_idxs[test_mask], labels[test_mask], node_labels[test_mask])

#   return full_data, node_features, edge_features, train_data, val_data, test_data




def get_data_ddp(
    dataset_name,
    different_new_nodes_between_val_and_test=False,
    randomize_features=False,
    use_validation=False,
):
    # 1. 加载数据
    # graph_df = pd.read_csv(f"./data/ml_{dataset_name}.csv")
    # edge_features = np.load(f"./data/ml_{dataset_name}.npy")
    # with open(f"./data/ml_{dataset_name}_node.pkl", "rb") as f:
    graph_df = pd.read_csv(r"/home/share/huadjyin/home/s_qinhua2/02code/tgn-master/spatiotemporal_mouse/data/ml_mouse_25_spatial.csv")
    #edge_features = np.load(f"./data/ml_{dataset_name}.npy")
    edge_features = np.load(r"/home/share/huadjyin/home/s_qinhua2/02code/tgn-master/spatiotemporal_mouse/data/ml_mouse_25_spatial.npy")
    # 读取 `node_features`
    #with open(f"./data/ml_{dataset_name}_node.pkl", "rb") as f:
    with open(r"/home/share/huadjyin/home/s_qinhua2/02code/tgn-master/spatiotemporal_mouse/data/ml_mouse_25_spatial_node.pkl", "rb") as f:
        node_features = pickle.load(f)

    # 2. 排序
    sorted_idx = np.argsort(graph_df.ts.values)
    graph_df = graph_df.iloc[sorted_idx].reset_index(drop=True)
    edge_features = edge_features[sorted_idx]

    if randomize_features:
        node_features = np.random.rand(*node_features.shape)

    # sources = graph_df.u.values
    # destinations = graph_df.i.values
    # edge_idxs = graph_df.idx.values - 1
    # labels = graph_df.label.values
    # timestamps = graph_df.ts.values
    sources = graph_df.u.values.astype(int)
    destinations = graph_df.i.values.astype(int)
    edge_idxs = (graph_df.idx.values - 1).astype(int)
    labels = graph_df.label.values.astype(int)  # 如果是整数 label（如分类）
    timestamps = graph_df.ts.values.astype(float)  # 若 ts 是 float 时间戳可保留 float

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    # 3. 按行数划分
    num = len(graph_df)
    train_end, val_end = int(0.70 * num), int(0.85 * num)
    train_mask_raw = np.arange(num) < train_end
    val_mask = (np.arange(num) >= train_end) & (np.arange(num) < val_end)
    test_mask = np.arange(num) >= val_end

    # 4. 新节点采样：仅从测试集采样
    random.seed(2025)
    test_sources = sources[test_mask]
    test_destinations = destinations[test_mask]
    test_node_set = set(test_sources) | set(test_destinations)

    all_nodes = set(sources) | set(destinations)
    sample_size = min(int(0.1 * len(all_nodes)), len(test_node_set))
    new_nodes = (
        set(random.sample(list(test_node_set), sample_size))
        if sample_size > 0
        else set()
    )

    # 5. 剔除训练集中涉及新节点的边
    u_mask = graph_df.u.map(lambda x: x in new_nodes).values
    i_mask = graph_df.i.map(lambda x: x in new_nodes).values
    observed_edges_mask = np.logical_and(~u_mask, ~i_mask)
    train_mask = np.logical_and(train_mask_raw, observed_edges_mask)

    # 6. 构造 Data 对象
    train_data = Data(
        sources[train_mask],
        destinations[train_mask],
        timestamps[train_mask],
        edge_idxs[train_mask],
        labels[train_mask],
    )
    val_data = Data(
        sources[val_mask],
        destinations[val_mask],
        timestamps[val_mask],
        edge_idxs[val_mask],
        labels[val_mask],
    )
    test_data = Data(
        sources[test_mask],
        destinations[test_mask],
        timestamps[test_mask],
        edge_idxs[test_mask],
        labels[test_mask],
    )

    # 7. 构造 new_node_val/test
    edge_contains_new_node_mask = np.array(
        [(u in new_nodes or v in new_nodes) for u, v in zip(sources, destinations)]
    )
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    new_node_val_data = Data(
        sources[new_node_val_mask],
        destinations[new_node_val_mask],
        timestamps[new_node_val_mask],
        edge_idxs[new_node_val_mask],
        labels[new_node_val_mask],
    )
    new_node_test_data = Data(
        sources[new_node_test_mask],
        destinations[new_node_test_mask],
        timestamps[new_node_test_mask],
        edge_idxs[new_node_test_mask],
        labels[new_node_test_mask],
    )

    # 8. 打印日志
    print(f"📊 数据统计：")
    print(
        f" - 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点"
    )
    print(
        f" - 验证集: {val_data.n_interactions} 交互, {val_data.n_unique_nodes} 个节点"
    )
    print(
        f" - 测试集: {test_data.n_interactions} 交互, {test_data.n_unique_nodes} 个节点"
    )
    print(f" - 新节点验证集: {new_node_val_data.n_interactions} 交互")
    print(f" - 新节点测试集: {new_node_test_data.n_interactions} 交互")

    return (
        full_data,
        node_features,
        edge_features,
        train_data,
        val_data,
        test_data,
        new_node_val_data,
        new_node_test_data,
    )


def get_data_link2(
    dataset_name,
    different_new_nodes_between_val_and_test=False,
    randomize_features=False,
    use_validation=False,
):
    # 1. 加载数据
    # graph_df = pd.read_csv(f"./data/ml_{dataset_name}.csv")
    # edge_features = np.load(f"./data/ml_{dataset_name}.npy")
    # with open(f"./data/ml_{dataset_name}_node.pkl", "rb") as f:
    graph_df = pd.read_csv(r"/home/share/huadjyin/home/s_qinhua2/02code/tgn-master/kidney/data/ml_aPT-B_3.csv")
    edge_features = np.load(r"/home/share/huadjyin/home/s_qinhua2/02code/tgn-master/kidney/data/ml_aPT-B_3.npy")
    with open(r"/home/share/huadjyin/home/s_qinhua2/02code/tgn-master/kidney/data/ml_aPT-B_3_node.pkl", "rb") as f:
        node_features = pickle.load(f)

    # 2. 排序
    sorted_idx = np.argsort(graph_df.ts.values)
    graph_df = graph_df.iloc[sorted_idx].reset_index(drop=True)
    edge_features = edge_features[sorted_idx]

    if randomize_features:
        node_features = np.random.rand(*node_features.shape)

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values - 1
    # labels = graph_df.label.values

    # ✅ 替换此处 label 的处理逻辑（强制转换为 float32 多标签矩阵）
    raw_labels = graph_df.label.values
    parsed_labels = [
        ast.literal_eval(x) if isinstance(x, str) else x for x in raw_labels
    ]
    labels = np.array(parsed_labels, dtype=np.float32)

    timestamps = graph_df.ts.values

    # Create full data object with all edges
    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    # Split by row count into train/val/test
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
    new_nodes = (
        set(random.sample(list(test_node_set), sample_size))
        if sample_size > 0
        else set()
    )

    # Remove edges in train set that involve new nodes
    u_mask = graph_df.u.map(lambda x: x in new_nodes).values
    i_mask = graph_df.i.map(lambda x: x in new_nodes).values
    observed_edges_mask = np.logical_and(~u_mask, ~i_mask)
    train_mask = np.logical_and(train_mask_raw, observed_edges_mask)

    # Build Data objects for each split
    train_data = Data(
        sources[train_mask],
        destinations[train_mask],
        timestamps[train_mask],
        edge_idxs[train_mask],
        labels[train_mask],
    )
    val_data = Data(
        sources[val_mask],
        destinations[val_mask],
        timestamps[val_mask],
        edge_idxs[val_mask],
        labels[val_mask],
    )
    test_data = Data(
        sources[test_mask],
        destinations[test_mask],
        timestamps[test_mask],
        edge_idxs[test_mask],
        labels[test_mask],
    )

    # 7. 构造 new_node_val/test
    edge_contains_new_node_mask = np.array(
        [(u in new_nodes or v in new_nodes) for u, v in zip(sources, destinations)]
    )
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    new_node_val_data = Data(
        sources[new_node_val_mask],
        destinations[new_node_val_mask],
        timestamps[new_node_val_mask],
        edge_idxs[new_node_val_mask],
        labels[new_node_val_mask],
    )
    new_node_test_data = Data(
        sources[new_node_test_mask],
        destinations[new_node_test_mask],
        timestamps[new_node_test_mask],
        edge_idxs[new_node_test_mask],
        labels[new_node_test_mask],
    )

    # 8. 打印日志
    print(f"📊 数据统计：")
    print(
        f" - 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点"
    )
    print(
        f" - 验证集: {val_data.n_interactions} 交互, {val_data.n_unique_nodes} 个节点"
    )
    print(
        f" - 测试集: {test_data.n_interactions} 交互, {test_data.n_unique_nodes} 个节点"
    )
    print(f" - 新节点验证集: {new_node_val_data.n_interactions} 交互")
    print(f" - 新节点测试集: {new_node_test_data.n_interactions} 交互")

    return (
        full_data,
        node_features,
        edge_features,
        train_data,
        val_data,
        test_data,
        new_node_val_data,
        new_node_test_data,
    )


def get_data_link1(
    dataset_name,
    different_new_nodes_between_val_and_test=False,
    randomize_features=False,
):

    # 1️⃣ 加载并排序
    # graph_df = pd.read_csv(f"./data/ml_{dataset_name}.csv")
    # edge_features = np.load(f"./data/ml_{dataset_name}.npy")
    # with open(f"./data/ml_{dataset_name}_node.pkl", "rb") as f:
    graph_df = pd.read_csv(r"/home/share/huadjyin/home/s_qinhua2/02code/tgn-master/kidney/data/ml_aPT-B_3.csv")
    edge_features = np.load(r"/home/share/huadjyin/home/s_qinhua2/02code/tgn-master/kidney/data/ml_aPT-B_3.npy")
    with open(r"/home/share/huadjyin/home/s_qinhua2/02code/tgn-master/kidney/data/ml_aPT-B_3_node.pkl", "rb") as f:
        node_features = pickle.load(f)

    sorted_indices = np.argsort(graph_df.ts.values)
    graph_df = graph_df.iloc[sorted_indices].reset_index(drop=True)
    edge_features = edge_features[sorted_indices]

    if randomize_features:
        node_features = np.random.rand(*node_features.shape)

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values - 1
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    # 2️⃣ 行数划分 train / val / test
    num = len(graph_df)
    train_end = int(0.70 * num)
    val_end = int(0.85 * num)
    train_mask_raw = np.arange(num) < train_end
    val_mask = (np.arange(num) >= train_end) & (np.arange(num) < val_end)
    test_mask = np.arange(num) >= val_end

    # 3️⃣ 采样新节点：仅从测试集中采样
    random.seed(2020)
    test_sources = sources[test_mask]
    test_destinations = destinations[test_mask]
    test_node_set = set(test_sources) | set(test_destinations)

    all_nodes = set(sources) | set(destinations)
    sample_size = min(int(0.1 * len(all_nodes)), len(test_node_set))
    new_test_node_set = (
        set(random.sample(list(test_node_set), sample_size))
        if sample_size > 0
        else set()
    )

    # 4️⃣ 剔除训练集中包含新节点的边
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
    observed_edges_mask = np.logical_and(
        ~new_test_source_mask, ~new_test_destination_mask
    )
    train_mask = np.logical_and(train_mask_raw, observed_edges_mask)

    # 5️⃣ 构建数据
    train_data = Data(
        sources[train_mask],
        destinations[train_mask],
        timestamps[train_mask],
        edge_idxs[train_mask],
        labels[train_mask],
    )
    val_data = Data(
        sources[val_mask],
        destinations[val_mask],
        timestamps[val_mask],
        edge_idxs[val_mask],
        labels[val_mask],
    )
    test_data = Data(
        sources[test_mask],
        destinations[test_mask],
        timestamps[test_mask],
        edge_idxs[test_mask],
        labels[test_mask],
    )

    # 6️⃣ 构建新节点验证集和测试集
    edge_contains_new_node_mask = np.array(
        [
            (a in new_test_node_set or b in new_test_node_set)
            for a, b in zip(sources, destinations)
        ]
    )
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    new_node_val_data = Data(
        sources[new_node_val_mask],
        destinations[new_node_val_mask],
        timestamps[new_node_val_mask],
        edge_idxs[new_node_val_mask],
        labels[new_node_val_mask],
    )
    new_node_test_data = Data(
        sources[new_node_test_mask],
        destinations[new_node_test_mask],
        timestamps[new_node_test_mask],
        edge_idxs[new_node_test_mask],
        labels[new_node_test_mask],
    )

    # 7️⃣ 打印日志
    print(f"📊 数据统计：")
    print(
        f" - 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点"
    )
    print(
        f" - 验证集: {val_data.n_interactions} 交互, {val_data.n_unique_nodes} 个节点"
    )
    print(
        f" - 测试集: {test_data.n_interactions} 交互, {test_data.n_unique_nodes} 个节点"
    )
    print(f" - 新节点验证集: {new_node_val_data.n_interactions} 交互")
    print(f" - 新节点测试集: {new_node_test_data.n_interactions} 交互")

    return (
        node_features,
        edge_features,
        full_data,
        train_data,
        val_data,
        test_data,
        new_node_val_data,
        new_node_test_data,
    )


def compute_time_statistics(sources, destinations, timestamps):
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

    return (
        mean_time_shift_src,
        std_time_shift_src,
        mean_time_shift_dst,
        std_time_shift_dst,
    )
