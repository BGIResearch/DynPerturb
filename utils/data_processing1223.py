import numpy as np
import pandas as pd
import random
from collections import defaultdict

class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels, celltypes=None):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.celltypes = celltypes
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)

def get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
    ### 加载数据并进行训练-验证-测试划分
    graph_df = pd.read_csv(f'./data/ml_{dataset_name}.csv')
    edge_features = np.load(f'./data/ml_{dataset_name}.npy')

    # 创建原始节点 ID 到连续索引的映射
    all_nodes = set(graph_df.u.values).union(set(graph_df.i.values))  # 合并 sources 和 destinations
    node_id_map = {node_id: idx for idx, node_id in enumerate(sorted(all_nodes))}
    print(f"Total nodes mapped: {len(node_id_map)}")
    print(f"Sample node mapping: {list(node_id_map.items())[:300]}")


    # 加载节点特征并转化为字典，键为连续的索引
    node_features = {}
    with open(f'./data/ml_{dataset_name}_node_features.csv', 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            parts = line.strip().split(',')
            node_id = int(parts[0])
            timestamp = float(parts[1])
            celltype = int(parts[2])
            features = np.array([float(x) for x in parts[3:]])
            # 将原始的节点 ID 映射为连续索引
            continuous_node_id = node_id_map[node_id]
            node_features[(continuous_node_id, timestamp, celltype)] = features

    if randomize_features:
        # 随机化节点特征，用于测试目的
        node_features = {key: np.random.rand(len(value)) for key, value in node_features.items()}
    
    print(f"Node feature tensor size: {len(node_features)}")
    # 定义训练、验证、测试的时间切分点
    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    # 将节点 ID 替换为连续索引
    sources = np.array([node_id_map[node_id] for node_id in graph_df.u.values])
    destinations = np.array([node_id_map[node_id] for node_id in graph_df.i.values])
    # sources = graph_df.u.values
    # destinations = graph_df.i.values
    timestamps = graph_df.ts.values
    labels = graph_df.label.values
    celltypes = graph_df.celltype.values  # 获取 celltype 信息
    edge_idxs = graph_df.idx.values
    

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels, celltypes)

    # 设置随机种子，确保可重复性
    random.seed(2020)

    # 创建节点集合并划分数据集
    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    # 确定测试集节点
    test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
    
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

    # 处理新测试节点的边的掩码
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # 训练集掩码（排除包含新测试节点的边）
    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask], celltypes[train_mask])

    # 定义新的节点集合
    train_node_set = set(train_data.sources).union(train_data.destinations)
    #assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set

    # 验证集和测试集的掩码
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

    # 创建验证集和测试集的新节点掩码
    if different_new_nodes_between_val_and_test:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
        test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

        edge_contains_new_val_node_mask = np.array(
            [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
        edge_contains_new_test_node_mask = np.array(
            [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
    else:
        edge_contains_new_node_mask = np.array(
            [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)
    # # 将节点 ID 替换为连续索引
    # unique_nodes = list(node_set)
    # node_id_map = {node_id: idx for idx, node_id in enumerate(unique_nodes)}

    # sources = np.array([node_id_map[node_id] for node_id in sources])
    # destinations = np.array([node_id_map[node_id] for node_id in destinations])

    # 创建数据集
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask], celltypes[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask], celltypes[test_mask])

    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask], edge_idxs[new_node_val_mask],
                             labels[new_node_val_mask], celltypes[new_node_val_mask])

    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                              labels[new_node_test_mask], celltypes[new_node_test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
        len(new_test_node_set)))

    return node_features, edge_features, full_data, train_data, val_data, test_data, \
           new_node_val_data, new_node_test_data,node_id_map




def get_data_node_classification(dataset_name, use_validation=False):
    ### Load data and train val test split
    graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
    edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))

    # node_features是一个字典，键为 (node_id, timestamp, celltype)，值为对应的特征列表
    node_features = defaultdict(list)
    with open(f'./data/ml_{dataset_name}_node_features.csv', 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            parts = line.strip().split(',')
            node_id = int(parts[0])
            timestamp = float(parts[1])
            celltype = int(parts[2])
            features = np.array([float(f) for f in parts[3:]])
            node_features[(node_id, timestamp, celltype)] = features

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values
    celltypes = graph_df.celltype.values  # 获取celltype信息

    random.seed(2020)

    # 计算train, validation, test时间范围
    train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    test_mask = timestamps > test_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels, celltypes)

    # 对训练集、验证集、测试集进行划分
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask], celltypes[train_mask])

    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask], celltypes[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask], celltypes[test_mask])

    return full_data, node_features, edge_features, train_data, val_data, test_data         

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

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst

