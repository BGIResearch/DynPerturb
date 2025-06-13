import numpy as np
import random
import pandas as pd
import pickle
import numpy as np
import ast
class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels,celltypes=None):
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


def get_data_node_classification(dataset_name, use_validation=False):
  ### Load data and train val test split
  graph_df = pd.read_csv(f'./data/ml_{dataset_name}.csv')
  edge_features = np.load(f'./data/ml_{dataset_name}.npy')

  # Read node features
  with open(f'./data/ml_{dataset_name}_node.pkl', 'rb') as f:
      node_features = pickle.load(f)

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values.astype(int)
  destinations = graph_df.i.values.astype(int)
  edge_idxs = graph_df.idx.values -1
  labels = graph_df.label.values  # For link prediction
#   node_labels = graph_df.node_label.values  # For node classification
  timestamps = graph_df.ts.values

  random.seed(2020)

  train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
  test_mask = timestamps > test_time
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  # Identify new nodes that only appear in validation and test datasets (inductive testing)
  node_set = set(sources) | set(destinations)
  train_node_set = set(train_data.sources) | set(train_data.destinations)

  # New nodes for validation and test (inductive testing)
  new_test_node_set = set(sources[test_mask]).union(set(destinations[test_mask]))
  new_val_node_set = set(sources[val_mask]).union(set(destinations[val_mask]))

  # Mask new node edges in validation and test data
  new_node_val_mask = np.isin(sources[val_mask], list(new_val_node_set)) | np.isin(destinations[val_mask], list(new_val_node_set))
  new_node_test_mask = np.isin(sources[test_mask], list(new_test_node_set)) | np.isin(destinations[test_mask], list(new_test_node_set))

  # new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
  #                          timestamps[new_node_val_mask],
  #                          edge_idxs[new_node_val_mask], labels[new_node_val_mask])

  # new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
  #                           timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
  #                           labels[new_node_test_mask])
  # return full_data, node_features, edge_features, train_data, val_data, test_data, new_node_val_data, new_node_test_data
  return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name)) 

  if randomize_features:
      node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values  # For link prediction task
  node_labels = graph_df.node_label.values  # For node classification task
  timestamps = graph_df.ts.values

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels, node_labels)

  random.seed(2020)

  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)

  # Compute nodes which appear at test time
  test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
  new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

  # Mask saying for each source and destination whether they are new test nodes
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask], node_labels[train_mask])

  # define the new nodes sets for testing inductiveness of the model
  train_node_set = set(train_data.sources).union(train_data.destinations)
  assert len(train_node_set & new_test_node_set) == 0
  new_node_set = node_set - train_node_set

  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time

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

  # validation and test with all edges
  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask], node_labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                    edge_idxs[test_mask], labels[test_mask], node_labels[test_mask])

  # validation and test with edges that at least has one new node (not in training set)
  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                            timestamps[new_node_val_mask], edge_idxs[new_node_val_mask],
                            labels[new_node_val_mask], node_labels[new_node_val_mask])

  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask], node_labels[new_node_test_mask])

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

  return node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data

def get_data_mulit(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False, use_validation=False):
  ### 1️⃣ **加载数据并按时间戳升序排序**
  graph_df = pd.read_csv(f'./data/ml_{dataset_name}.csv')
  edge_features = np.load(f'./data/ml_{dataset_name}.npy')

  # 读取 `node_features`
  with open(f'./data/ml_{dataset_name}_node.pkl', 'rb') as f:
      node_features = pickle.load(f)

  # **按时间戳升序排序**
  sorted_indices = np.argsort(graph_df.ts.values)
  graph_df = graph_df.iloc[sorted_indices].reset_index(drop=True)
  edge_features = edge_features[sorted_indices]

  # **随机初始化节点特征（可选）**
  if randomize_features:
      node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

  # 提取字段
  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values - 1
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  # **计算训练集、验证集、测试集的分割时间戳**
  val_time, test_time = np.quantile(graph_df.ts, [0.70, 0.85])

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  random.seed(2020)
  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)

  # **计算测试集中新节点**
  test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
  num_sample = int(0.1 * n_total_unique_nodes)
  num_sample = min(num_sample, len(test_node_set))  # 防止超出 test_node_set 大小

  if num_sample > 0:
      new_test_node_set = set(random.sample(test_node_set, num_sample))
  else:
      new_test_node_set = set()

  # new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

  # **为每个源和目标节点创建是否为新测试节点的标记**
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

  # **划分训练集**
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  # **为验证集和测试集创建数据**
  val_mask = (timestamps <= test_time) & (timestamps > val_time)
  test_mask = timestamps > test_time

  # **为新节点创建验证集和测试集**
  edge_contains_new_node_mask = np.array([(a in new_test_node_set or b in new_test_node_set) for a, b in zip(sources, destinations)])
  new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
  new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                    edge_idxs[test_mask], labels[test_mask])

  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                            timestamps[new_node_val_mask], edge_idxs[new_node_val_mask], labels[new_node_val_mask])

  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask], labels[new_node_test_mask])

  print(f"📊 数据统计：")


  print(f" - 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点")
  print(f" - 验证集: {val_data.n_interactions} 交互, {val_data.n_unique_nodes} 个节点")
  print(f" - 测试集: {test_data.n_interactions} 交互, {test_data.n_unique_nodes} 个节点")

  # return node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data
  return full_data, node_features, edge_features, train_data, val_data, test_data, new_node_val_data, new_node_test_data

def get_data_mulit_0423(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False, use_validation=False):
    # 1. 加载数据
    graph_df = pd.read_csv(f'./data/ml_{dataset_name}.csv')
    edge_features = np.load(f'./data/ml_{dataset_name}.npy')
    with open(f'./data/ml_{dataset_name}_node.pkl', 'rb') as f:
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
    test_sources = sources[test_mask]
    test_destinations = destinations[test_mask]
    test_node_set = set(test_sources) | set(test_destinations)

    all_nodes = set(sources) | set(destinations)
    sample_size = min(int(0.1 * len(all_nodes)), len(test_node_set))
    new_nodes = set(random.sample(list(test_node_set), sample_size)) if sample_size > 0 else set()

    # 5. 剔除训练集中涉及新节点的边
    u_mask = graph_df.u.map(lambda x: x in new_nodes).values
    i_mask = graph_df.i.map(lambda x: x in new_nodes).values
    observed_edges_mask = np.logical_and(~u_mask, ~i_mask)
    train_mask = np.logical_and(train_mask_raw, observed_edges_mask)

    # 6. 构造 Data 对象
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    # 7. 构造 new_node_val/test
    edge_contains_new_node_mask = np.array([(u in new_nodes or v in new_nodes) for u, v in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask], edge_idxs[new_node_val_mask], labels[new_node_val_mask])
    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask], labels[new_node_test_mask])

    # 8. 打印日志
    print(f"📊 数据统计：")
    print(f" - 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点")
    print(f" - 验证集: {val_data.n_interactions} 交互, {val_data.n_unique_nodes} 个节点")
    print(f" - 测试集: {test_data.n_interactions} 交互, {test_data.n_unique_nodes} 个节点")
    print(f" - 新节点验证集: {new_node_val_data.n_interactions} 交互")
    print(f" - 新节点测试集: {new_node_test_data.n_interactions} 交互")

    return full_data, node_features, edge_features, train_data, val_data, test_data, new_node_val_data, new_node_test_data


def get_data_mulit_0512(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False, use_validation=False):
    # 1. 加载数据
    graph_df = pd.read_csv(f'./data/ml_{dataset_name}.csv')
    edge_features = np.load(f'./data/ml_{dataset_name}.npy')
    with open(f'./data/ml_{dataset_name}_node.pkl', 'rb') as f:
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
    parsed_labels = [ast.literal_eval(x) if isinstance(x, str) else x for x in raw_labels]
    labels = np.array(parsed_labels, dtype=np.float32)
    
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    # 3. 按行数划分
    num = len(graph_df)
    train_end, val_end = int(0.70 * num), int(0.85 * num)
    train_mask_raw = np.arange(num) < train_end
    val_mask = (np.arange(num) >= train_end) & (np.arange(num) < val_end)
    test_mask = np.arange(num) >= val_end

    # 4. 新节点采样：仅从测试集采样
    test_sources = sources[test_mask]
    test_destinations = destinations[test_mask]
    test_node_set = set(test_sources) | set(test_destinations)

    all_nodes = set(sources) | set(destinations)
    sample_size = min(int(0.1 * len(all_nodes)), len(test_node_set))
    new_nodes = set(random.sample(list(test_node_set), sample_size)) if sample_size > 0 else set()

    # 5. 剔除训练集中涉及新节点的边
    u_mask = graph_df.u.map(lambda x: x in new_nodes).values
    i_mask = graph_df.i.map(lambda x: x in new_nodes).values
    observed_edges_mask = np.logical_and(~u_mask, ~i_mask)
    train_mask = np.logical_and(train_mask_raw, observed_edges_mask)

    # 6. 构造 Data 对象
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    # 7. 构造 new_node_val/test
    edge_contains_new_node_mask = np.array([(u in new_nodes or v in new_nodes) for u, v in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask], edge_idxs[new_node_val_mask], labels[new_node_val_mask])
    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask], labels[new_node_test_mask])

    # 8. 打印日志
    print(f"📊 数据统计：")
    print(f" - 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点")
    print(f" - 验证集: {val_data.n_interactions} 交互, {val_data.n_unique_nodes} 个节点")
    print(f" - 测试集: {test_data.n_interactions} 交互, {test_data.n_unique_nodes} 个节点")
    print(f" - 新节点验证集: {new_node_val_data.n_interactions} 交互")
    print(f" - 新节点测试集: {new_node_test_data.n_interactions} 交互")

    return full_data, node_features, edge_features, train_data, val_data, test_data, new_node_val_data, new_node_test_data


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
