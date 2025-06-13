import numpy as np
import random
import pandas as pd
import pickle


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


def get_data_node_classification(dataset_name, use_validation=False):
  ### Load data and train val test split
  graph_df = pd.read_csv(f'./data/ml_{dataset_name}.csv')
  edge_features = np.load(f'./data/ml_{dataset_name}.npy')

  # 读取 `node_features`
  with open(f'./data/ml_{dataset_name}_node.pkl', 'rb') as f:
      node_features = pickle.load(f)

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  # sources = graph_df.u.values
  # destinations = graph_df.i.values
  sources = graph_df.u.values.astype(int)  # 强制转换为整型
  destinations = graph_df.i.values.astype(int)  # 强制转换为整型
  edge_idxs = graph_df.idx.values - 1
  labels = graph_df.label.values
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

  return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data_node_classification_0423(dataset_name, use_validation=False):
    # === 1. 加载数据并按时间戳升序排序 ===
    graph_df = pd.read_csv(f'./data/ml_{dataset_name}.csv')
    edge_features = np.load(f'./data/ml_{dataset_name}.npy')
    with open(f'./data/ml_{dataset_name}_node.pkl', 'rb') as f:
        node_features = pickle.load(f)

    sorted_indices = np.argsort(graph_df.ts.values)
    graph_df = graph_df.iloc[sorted_indices].reset_index(drop=True)
    edge_features = edge_features[sorted_indices]

    # === 2. 提取字段 ===
    sources = graph_df.u.values.astype(int)
    destinations = graph_df.i.values.astype(int)
    edge_idxs = graph_df.idx.values - 1
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    # === 3. 按行数划分 ===
    num_interactions = len(graph_df)
    train_end = int(0.70 * num_interactions)
    val_end = int(0.85 * num_interactions)

    train_mask_raw = np.arange(num_interactions) < train_end
    val_mask = (np.arange(num_interactions) >= train_end) & (np.arange(num_interactions) < val_end)
    test_mask = np.arange(num_interactions) >= val_end

    # === 4. 抽取新节点并从训练集中剔除相关边 ===
    random.seed(2020)
    node_set = set(sources) | set(destinations)
    test_sources = sources[test_mask]
    test_destinations = destinations[test_mask]
    test_node_set = set(test_sources) | set(test_destinations)
    num_sample = min(int(0.1 * len(node_set)), len(test_node_set))
    new_test_node_set = set(random.sample(list(test_node_set), num_sample)) if num_sample > 0 else set()

    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # === 5. 最终训练集（剔除新节点相关边） ===
    train_mask = np.logical_and(train_mask_raw, observed_edges_mask)

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  #node_features = pickle.load('./data/ml_{}_node.pkl'.format(dataset_name)) 
    # 确保时间戳列按升序排序
  sorted_indices = np.argsort(graph_df.ts.values)  # 获取排序索引

  # 使用排序索引对 `graph_df` 和 `edge_features` 进行同步排序
  graph_df = graph_df.iloc[sorted_indices].reset_index(drop=True)
  edge_features = edge_features[sorted_indices]
  with open('./data/ml_{}_node.pkl'.format(dataset_name), 'rb') as f:
    node_features = pickle.load(f)

  if randomize_features:
    node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values-1
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  random.seed(2020)

  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)

  # Compute nodes which appear at test time
  test_node_set = set(sources[timestamps > val_time]).union(
    set(destinations[timestamps > val_time]))
  # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
  # their edges from training
  new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))
  #   # **计算新测试节点数**
  # num_test_nodes = min(len(test_node_set), int(0.1 * n_total_unique_nodes))  # 限制最大数量

  # # **如果 `num_test_nodes` 为 0，避免 `random.sample()` 报错**
  # if num_test_nodes > 0:
  #     new_test_node_set = set(random.sample(test_node_set, num_test_nodes))
  # else:
  #     new_test_node_set = set()  # 无法取样，返回空集合

  # # **打印调试信息**
  # print(f"📌 Debug: test_node_set={len(test_node_set)}, n_total_unique_nodes={n_total_unique_nodes}, num_test_nodes={num_test_nodes}")

  # Mask saying for each source and destination whether they are new test nodes
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  # Mask which is true for edges with both destination and source not being new test nodes (because
  # we want to remove all edges involving any new test node)
  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

  # For train we keep edges happening before the validation time which do not involve any new node
  # used for inductiveness
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

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
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  # validation and test with edges that at least has one new node (not in training set)
  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])

  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])

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
         new_node_val_data, new_node_test_data



def get_data_0313(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
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

    # from data_structures import Data  # 你的 `Data` 类定义

    ### 2️⃣ **按 70% / 15% / 15% 分割数据**
    n_interactions = len(graph_df)
    train_end = int(0.70 * n_interactions)  # 前 70%
    val_end = int(0.85 * n_interactions)    # 70% ~ 85%

    # **创建 mask**
    train_mask = np.arange(n_interactions) < train_end
    val_mask = (np.arange(n_interactions) >= train_end) & (np.arange(n_interactions) < val_end)
    test_mask = np.arange(n_interactions) >= val_end

    # **创建 `Data` 对象**
    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    print(f"📊 数据统计：")
    print(f" - 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点")
    print(f" - 验证集: {val_data.n_interactions} 交互, {val_data.n_unique_nodes} 个节点")
    print(f" - 测试集: {test_data.n_interactions} 交互, {test_data.n_unique_nodes} 个节点")

    ### 3️⃣ **从 `val + test` 中选取 `10%` 作为 `new_test_node_set`**
    random.seed(2020)

    train_node_set = set(train_data.sources).union(train_data.destinations)
    val_test_node_set = set(val_data.sources).union(val_data.destinations, 
                                                    test_data.sources, 
                                                    test_data.destinations)
    node_set = train_node_set.union(val_test_node_set)
    n_total_unique_nodes = len(node_set)

    # **新节点 = `val + test` 里 10% 的随机抽样**
    num_test_nodes = min(len(val_test_node_set), int(0.1 * n_total_unique_nodes))
    new_test_node_set = set(random.sample(list(val_test_node_set), num_test_nodes)) if num_test_nodes > 0 else set()

    print(f"📌 Debug: test_node_set={len(val_test_node_set)}, n_total_unique_nodes={n_total_unique_nodes}, num_test_nodes={num_test_nodes}")

    ### 4️⃣ **删除 `训练集` 中包含 `new_test_node_set` 的边**
    new_test_source_mask = np.array([src in new_test_node_set for src in train_data.sources])
    new_test_destination_mask = np.array([dst in new_test_node_set for dst in train_data.destinations])
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    train_data = Data(
        train_data.sources[observed_edges_mask],
        train_data.destinations[observed_edges_mask],
        train_data.timestamps[observed_edges_mask],
        train_data.edge_idxs[observed_edges_mask],
        train_data.labels[observed_edges_mask],
    )
    print(f"📌 剔除新节点后, 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点")

    ### 5️⃣ **划分 `new_node_val_data` 和 `new_node_test_data`**
    new_node_set = node_set - train_node_set  # 训练集中未出现的新节点

    # **检查 `val/test` 里哪些边含有 `new_node_set`**
    edge_contains_new_val_node_mask = np.array([(a in new_node_set or b in new_node_set)
                                                 for a, b in zip(val_data.sources, val_data.destinations)])
    edge_contains_new_test_node_mask = np.array([(a in new_node_set or b in new_node_set)
                                                 for a, b in zip(test_data.sources, test_data.destinations)])

    new_node_val_data = Data(val_data.sources[edge_contains_new_val_node_mask],
                             val_data.destinations[edge_contains_new_val_node_mask],
                             val_data.timestamps[edge_contains_new_val_node_mask],
                             val_data.edge_idxs[edge_contains_new_val_node_mask],
                             val_data.labels[edge_contains_new_val_node_mask])

    new_node_test_data = Data(test_data.sources[edge_contains_new_test_node_mask],
                              test_data.destinations[edge_contains_new_test_node_mask],
                              test_data.timestamps[edge_contains_new_test_node_mask],
                              test_data.edge_idxs[edge_contains_new_test_node_mask],
                              test_data.labels[edge_contains_new_test_node_mask])

    ### 6️⃣ **打印最终数据统计**
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

def get_data_0422(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
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

    # from data_structures import Data  # 你的 `Data` 类定义

    # **计算训练集、验证集、测试集的分割时间戳**
    val_time, test_time = np.quantile(graph_df.ts, [0.60, 0.75])

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    random.seed(2020)
    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    # **计算测试集中新节点**
    test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

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
    return node_features, edge_features,full_data,train_data, val_data, test_data, new_node_val_data, new_node_test_data

def get_data_0422_70_15_15(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
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

    # from data_structures import Data  # 你的 `Data` 类定义

    # **计算训练集、验证集、测试集的分割时间戳**
    val_time, test_time = np.quantile(graph_df.ts, [0.70, 0.85])

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    random.seed(2020)
    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    # **计算测试集中新节点**
    test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

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
    return node_features, edge_features,full_data,train_data, val_data, test_data, new_node_val_data, new_node_test_data

def get_data_0423_60_15_25(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):


    # 1️⃣ 加载并排序
    graph_df = pd.read_csv(f'./data/ml_{dataset_name}.csv')
    edge_features = np.load(f'./data/ml_{dataset_name}.npy')
    with open(f'./data/ml_{dataset_name}_node.pkl', 'rb') as f:
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
    train_end = int(0.60 * num)
    val_end = int(0.75 * num)
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
    new_test_node_set = set(random.sample(list(test_node_set), sample_size)) if sample_size > 0 else set()

    # 4️⃣ 剔除训练集中包含新节点的边
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
    train_mask = np.logical_and(train_mask_raw, observed_edges_mask)

    # 5️⃣ 构建数据
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    # 6️⃣ 构建新节点验证集和测试集
    edge_contains_new_node_mask = np.array([(a in new_test_node_set or b in new_test_node_set)
                                            for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask], edge_idxs[new_node_val_mask],
                             labels[new_node_val_mask])
    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                              labels[new_node_test_mask])

    # 7️⃣ 打印日志
    print(f"📊 数据统计：")
    print(f" - 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点")
    print(f" - 验证集: {val_data.n_interactions} 交互, {val_data.n_unique_nodes} 个节点")
    print(f" - 测试集: {test_data.n_interactions} 交互, {test_data.n_unique_nodes} 个节点")
    print(f" - 新节点验证集: {new_node_val_data.n_interactions} 交互")
    print(f" - 新节点测试集: {new_node_test_data.n_interactions} 交互")

    return node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data

def get_data_0423_70_15_15(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):

    # 1️⃣ 加载并排序
    graph_df = pd.read_csv(f'./data/ml_{dataset_name}.csv')
    edge_features = np.load(f'./data/ml_{dataset_name}.npy')
    with open(f'./data/ml_{dataset_name}_node.pkl', 'rb') as f:
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
    new_test_node_set = set(random.sample(list(test_node_set), sample_size)) if sample_size > 0 else set()

    # 4️⃣ 剔除训练集中包含新节点的边
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
    train_mask = np.logical_and(train_mask_raw, observed_edges_mask)

    # 5️⃣ 构建数据
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    # 6️⃣ 构建新节点验证集和测试集
    edge_contains_new_node_mask = np.array([(a in new_test_node_set or b in new_test_node_set)
                                            for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask], edge_idxs[new_node_val_mask],
                             labels[new_node_val_mask])
    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                              labels[new_node_test_mask])

    # 7️⃣ 打印日志
    print(f"📊 数据统计：")
    print(f" - 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点")
    print(f" - 验证集: {val_data.n_interactions} 交互, {val_data.n_unique_nodes} 个节点")
    print(f" - 测试集: {test_data.n_interactions} 交互, {test_data.n_unique_nodes} 个节点")
    print(f" - 新节点验证集: {new_node_val_data.n_interactions} 交互")
    print(f" - 新节点测试集: {new_node_test_data.n_interactions} 交互")

    return node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data


def get_data_0313_change(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
    ### 1️⃣ **加载数据并按时间戳升序排序**
    graph_df = pd.read_csv(f'./data/ml_{dataset_name}.csv')
    edge_features = np.load(f'./data/ml_{dataset_name}.npy')

    # 读取 `node_features`
    with open(f'./data/ml_{dataset_name}_change_node.pkl', 'rb') as f:
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

    # from data_structures import Data  # 你的 `Data` 类定义

    ### 2️⃣ **按 70% / 15% / 15% 分割数据**
    n_interactions = len(graph_df)
    train_end = int(0.70 * n_interactions)  # 前 70%
    val_end = int(0.85 * n_interactions)    # 70% ~ 85%

    # **创建 mask**
    train_mask = np.arange(n_interactions) < train_end
    val_mask = (np.arange(n_interactions) >= train_end) & (np.arange(n_interactions) < val_end)
    test_mask = np.arange(n_interactions) >= val_end

    # **创建 `Data` 对象**
    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    print(f"📊 数据统计：")
    print(f" - 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点")
    print(f" - 验证集: {val_data.n_interactions} 交互, {val_data.n_unique_nodes} 个节点")
    print(f" - 测试集: {test_data.n_interactions} 交互, {test_data.n_unique_nodes} 个节点")

    ### 3️⃣ **从 `val + test` 中选取 `10%` 作为 `new_test_node_set`**
    random.seed(2020)

    train_node_set = set(train_data.sources).union(train_data.destinations)
    val_test_node_set = set(val_data.sources).union(val_data.destinations, 
                                                    test_data.sources, 
                                                    test_data.destinations)
    node_set = train_node_set.union(val_test_node_set)
    n_total_unique_nodes = len(node_set)

    # **新节点 = `val + test` 里 10% 的随机抽样**
    num_test_nodes = min(len(val_test_node_set), int(0.1 * n_total_unique_nodes))
    new_test_node_set = set(random.sample(list(val_test_node_set), num_test_nodes)) if num_test_nodes > 0 else set()

    print(f"📌 Debug: test_node_set={len(val_test_node_set)}, n_total_unique_nodes={n_total_unique_nodes}, num_test_nodes={num_test_nodes}")

    ### 4️⃣ **删除 `训练集` 中包含 `new_test_node_set` 的边**
    new_test_source_mask = np.array([src in new_test_node_set for src in train_data.sources])
    new_test_destination_mask = np.array([dst in new_test_node_set for dst in train_data.destinations])
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    train_data = Data(
        train_data.sources[observed_edges_mask],
        train_data.destinations[observed_edges_mask],
        train_data.timestamps[observed_edges_mask],
        train_data.edge_idxs[observed_edges_mask],
        train_data.labels[observed_edges_mask],
    )
    print(f"📌 剔除新节点后, 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点")

    ### 5️⃣ **划分 `new_node_val_data` 和 `new_node_test_data`**
    new_node_set = node_set - train_node_set  # 训练集中未出现的新节点

    # **检查 `val/test` 里哪些边含有 `new_node_set`**
    edge_contains_new_val_node_mask = np.array([(a in new_node_set or b in new_node_set)
                                                 for a, b in zip(val_data.sources, val_data.destinations)])
    edge_contains_new_test_node_mask = np.array([(a in new_node_set or b in new_node_set)
                                                 for a, b in zip(test_data.sources, test_data.destinations)])

    new_node_val_data = Data(val_data.sources[edge_contains_new_val_node_mask],
                             val_data.destinations[edge_contains_new_val_node_mask],
                             val_data.timestamps[edge_contains_new_val_node_mask],
                             val_data.edge_idxs[edge_contains_new_val_node_mask],
                             val_data.labels[edge_contains_new_val_node_mask])

    new_node_test_data = Data(test_data.sources[edge_contains_new_test_node_mask],
                              test_data.destinations[edge_contains_new_test_node_mask],
                              test_data.timestamps[edge_contains_new_test_node_mask],
                              test_data.edge_idxs[edge_contains_new_test_node_mask],
                              test_data.labels[edge_contains_new_test_node_mask])

    ### 6️⃣ **打印最终数据统计**
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

def get_data_0408(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
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

    # from data_structures import Data  # 你的 `Data` 类定义

    ### 2️⃣ **按 70% / 15% / 15% 分割数据**
    n_interactions = len(graph_df)
    train_end = int(0.70 * n_interactions)  # 前 70%
    val_end = int(0.85 * n_interactions)    # 70% ~ 85%

    # **创建 mask**
    train_mask = np.arange(n_interactions) < train_end
    val_mask = (np.arange(n_interactions) >= train_end) & (np.arange(n_interactions) < val_end)
    test_mask = np.arange(n_interactions) >= val_end

    # **创建 `Data` 对象**
    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    print(f"📊 数据统计：")
    print(f" - 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点")
    print(f" - 验证集: {val_data.n_interactions} 交互, {val_data.n_unique_nodes} 个节点")
    print(f" - 测试集: {test_data.n_interactions} 交互, {test_data.n_unique_nodes} 个节点")

    ### 3️⃣ **从 `val + test` 中选取 `10%` 作为 `new_test_node_set`**
    random.seed(2020)

    train_node_set = set(train_data.sources).union(train_data.destinations)
    val_test_node_set = set(val_data.sources).union(val_data.destinations, 
                                                    test_data.sources, 
                                                    test_data.destinations)
    node_set = train_node_set.union(val_test_node_set)
    n_total_unique_nodes = len(node_set)

    # **新节点 = `val + test` 里 10% 的随机抽样**
    num_test_nodes = min(len(val_test_node_set), int(0.1 * n_total_unique_nodes))
    new_test_node_set = set(random.sample(list(val_test_node_set), num_test_nodes)) if num_test_nodes > 0 else set()

    print(f"📌 Debug: test_node_set={len(val_test_node_set)}, n_total_unique_nodes={n_total_unique_nodes}, num_test_nodes={num_test_nodes}")

    ### 4️⃣ **删除 `训练集` 中包含 `new_test_node_set` 的边**
    new_test_source_mask = np.array([src in new_test_node_set for src in train_data.sources])
    new_test_destination_mask = np.array([dst in new_test_node_set for dst in train_data.destinations])
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    train_data = Data(
        train_data.sources[observed_edges_mask],
        train_data.destinations[observed_edges_mask],
        train_data.timestamps[observed_edges_mask],
        train_data.edge_idxs[observed_edges_mask],
        train_data.labels[observed_edges_mask],
    )
    print(f"📌 剔除新节点后, 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点")
    # ✅ 重新对 train_data 按照时间戳升序排序
    sorted_idx = np.argsort(train_data.timestamps)
    train_data = Data(
        train_data.sources[sorted_idx],
        train_data.destinations[sorted_idx],
        train_data.timestamps[sorted_idx],
        train_data.edge_idxs[sorted_idx],
        train_data.labels[sorted_idx],
    )
    print("✅ 训练集已按时间戳重新排序。")

    ### 5️⃣ **划分 `new_node_val_data` 和 `new_node_test_data`**
    new_node_set = node_set - train_node_set  # 训练集中未出现的新节点

    # **检查 `val/test` 里哪些边含有 `new_node_set`**
    edge_contains_new_val_node_mask = np.array([(a in new_node_set or b in new_node_set)
                                                 for a, b in zip(val_data.sources, val_data.destinations)])
    edge_contains_new_test_node_mask = np.array([(a in new_node_set or b in new_node_set)
                                                 for a, b in zip(test_data.sources, test_data.destinations)])

    new_node_val_data = Data(val_data.sources[edge_contains_new_val_node_mask],
                             val_data.destinations[edge_contains_new_val_node_mask],
                             val_data.timestamps[edge_contains_new_val_node_mask],
                             val_data.edge_idxs[edge_contains_new_val_node_mask],
                             val_data.labels[edge_contains_new_val_node_mask])

    new_node_test_data = Data(test_data.sources[edge_contains_new_test_node_mask],
                              test_data.destinations[edge_contains_new_test_node_mask],
                              test_data.timestamps[edge_contains_new_test_node_mask],
                              test_data.edge_idxs[edge_contains_new_test_node_mask],
                              test_data.labels[edge_contains_new_test_node_mask])

    ### 6️⃣ **打印最终数据统计**
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

def get_data_0314(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
    ### 1️⃣ **加载数据并按时间戳升序排序**
    graph_df = pd.read_csv(f'./data_new/ml_{dataset_name}.csv')
    edge_features = np.load(f'./data_new/ml_{dataset_name}.npy')

    # 读取 `node_features`
    with open(f'./data_new/ml_{dataset_name}_node.pkl', 'rb') as f:
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

    # from data_structures import Data  # 你的 `Data` 类定义

    ### 2️⃣ **按 70% / 15% / 15% 分割数据**
    n_interactions = len(graph_df)
    train_end = int(0.70 * n_interactions)  # 前 70%
    val_end = int(0.85 * n_interactions)    # 70% ~ 85%

    # **创建 mask**
    train_mask = np.arange(n_interactions) < train_end
    val_mask = (np.arange(n_interactions) >= train_end) & (np.arange(n_interactions) < val_end)
    test_mask = np.arange(n_interactions) >= val_end

    # **创建 `Data` 对象**
    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    print(f"📊 数据统计：")
    print(f" - 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点")
    print(f" - 验证集: {val_data.n_interactions} 交互, {val_data.n_unique_nodes} 个节点")
    print(f" - 测试集: {test_data.n_interactions} 交互, {test_data.n_unique_nodes} 个节点")

    ### 3️⃣ **从 `val + test` 中选取 `10%` 作为 `new_test_node_set`**
    random.seed(2020)

    train_node_set = set(train_data.sources).union(train_data.destinations)
    val_test_node_set = set(val_data.sources).union(val_data.destinations, 
                                                    test_data.sources, 
                                                    test_data.destinations)
    node_set = train_node_set.union(val_test_node_set)
    n_total_unique_nodes = len(node_set)

    # **新节点 = `val + test` 里 10% 的随机抽样**
    num_test_nodes = min(len(val_test_node_set), int(0.1 * n_total_unique_nodes))
    new_test_node_set = set(random.sample(list(val_test_node_set), num_test_nodes)) if num_test_nodes > 0 else set()

    print(f"📌 Debug: test_node_set={len(val_test_node_set)}, n_total_unique_nodes={n_total_unique_nodes}, num_test_nodes={num_test_nodes}")

    ### 4️⃣ **删除 `训练集` 中包含 `new_test_node_set` 的边**
    new_test_source_mask = np.array([src in new_test_node_set for src in train_data.sources])
    new_test_destination_mask = np.array([dst in new_test_node_set for dst in train_data.destinations])
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    train_data = Data(
        train_data.sources[observed_edges_mask],
        train_data.destinations[observed_edges_mask],
        train_data.timestamps[observed_edges_mask],
        train_data.edge_idxs[observed_edges_mask],
        train_data.labels[observed_edges_mask],
    )
    print(f"📌 剔除新节点后, 训练集: {train_data.n_interactions} 交互, {train_data.n_unique_nodes} 个节点")

    ### 5️⃣ **划分 `new_node_val_data` 和 `new_node_test_data`**
    new_node_set = node_set - train_node_set  # 训练集中未出现的新节点

    # **检查 `val/test` 里哪些边含有 `new_node_set`**
    edge_contains_new_val_node_mask = np.array([(a in new_node_set or b in new_node_set)
                                                 for a, b in zip(val_data.sources, val_data.destinations)])
    edge_contains_new_test_node_mask = np.array([(a in new_node_set or b in new_node_set)
                                                 for a, b in zip(test_data.sources, test_data.destinations)])

    new_node_val_data = Data(val_data.sources[edge_contains_new_val_node_mask],
                             val_data.destinations[edge_contains_new_val_node_mask],
                             val_data.timestamps[edge_contains_new_val_node_mask],
                             val_data.edge_idxs[edge_contains_new_val_node_mask],
                             val_data.labels[edge_contains_new_val_node_mask])

    new_node_test_data = Data(test_data.sources[edge_contains_new_test_node_mask],
                              test_data.destinations[edge_contains_new_test_node_mask],
                              test_data.timestamps[edge_contains_new_test_node_mask],
                              test_data.edge_idxs[edge_contains_new_test_node_mask],
                              test_data.labels[edge_contains_new_test_node_mask])

    ### 6️⃣ **打印最终数据统计**
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

def get_data_copy(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/all_data/ml_all_data_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/all_data/ml_all_data_{}.npy'.format(dataset_name))
  #node_features = pickle.load('./data/ml_{}_node.pkl'.format(dataset_name)) 
    # 确保时间戳列按升序排序
  sorted_indices = np.argsort(graph_df.ts.values)  # 获取排序索引

  # 使用排序索引对 `graph_df` 和 `edge_features` 进行同步排序
  graph_df = graph_df.iloc[sorted_indices].reset_index(drop=True)
  edge_features = edge_features[sorted_indices]
  with open('./data/all_data/ml_all_data_{}_node copy.pkl'.format(dataset_name), 'rb') as f:
    node_features = pickle.load(f)

  if randomize_features:
    node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values-1
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  random.seed(2020)

  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)

  # Compute nodes which appear at test time
  test_node_set = set(sources[timestamps > val_time]).union(
    set(destinations[timestamps > val_time]))
  # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
  # their edges from training
  new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

  # Mask saying for each source and destination whether they are new test nodes
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  # Mask which is true for edges with both destination and source not being new test nodes (because
  # we want to remove all edges involving any new test node)
  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

  # For train we keep edges happening before the validation time which do not involve any new node
  # used for inductiveness
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

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
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  # validation and test with edges that at least has one new node (not in training set)
  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])

  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])

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
         new_node_val_data, new_node_test_data

def modify_node_feature(node_features, node_to_modify, modify_time):
    """
    修改特定节点在特定时间戳的特征为零。
    
    :param node_features: 包含所有节点特征的字典，结构为 {node_id: {time: feature_vector}}。
    :param node_to_modify: 需要修改特征的节点ID。
    :param modify_time: 需要修改的时间戳。
    """
    if node_to_modify in node_features:
        # 检查该节点是否在指定时间戳上有特征
        if modify_time in node_features[node_to_modify]:
            node_features[node_to_modify][modify_time] = np.zeros_like(node_features[node_to_modify][modify_time])
            print(f"Node {node_to_modify}'s feature at time {modify_time} has been set to zero.")
        else:
            print(f"Node {node_to_modify} does not have a feature at time {modify_time}.")
    else:
        print(f"Node {node_to_modify} is not in the dataset.")
    
    return node_features


def get_and_change_data(dataset_name,node_to_modify=None,modify_time=None, different_new_nodes_between_val_and_test=False, randomize_features=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('./data/all_data/ml_all_data_{}.csv'.format(dataset_name))
  edge_features = np.load('./data/all_data/ml_all_data_{}.npy'.format(dataset_name))
  #node_features = pickle.load('./data/ml_{}_node.pkl'.format(dataset_name)) 
    # 确保时间戳列按升序排序
  sorted_indices = np.argsort(graph_df.ts.values)  # 获取排序索引

  # 使用排序索引对 `graph_df` 和 `edge_features` 进行同步排序
  graph_df = graph_df.iloc[sorted_indices].reset_index(drop=True)
  edge_features = edge_features[sorted_indices]
  with open('./data/all_data/ml_all_data_{}_node.pkl'.format(dataset_name), 'rb') as f:
    node_features = pickle.load(f)

  if randomize_features:
    node_features = np.random.rand(node_features.shape[0], node_features.shape[1])
  
  #修改指定节点指定时间得初始特征，默认修改为0  
  if node_to_modify is not None and modify_time is not None:
      node_features = modify_node_feature(node_features, node_to_modify, modify_time)


  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values-1
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  random.seed(2020)

  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)

  # Compute nodes which appear at test time
  test_node_set = set(sources[timestamps > val_time]).union(
    set(destinations[timestamps > val_time]))
  # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
  # their edges from training
  new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

  # Mask saying for each source and destination whether they are new test nodes
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  # Mask which is true for edges with both destination and source not being new test nodes (because
  # we want to remove all edges involving any new test node)
  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

  # For train we keep edges happening before the validation time which do not involve any new node
  # used for inductiveness
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

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
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  # validation and test with edges that at least has one new node (not in training set)
  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])

  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])

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
         new_node_val_data, new_node_test_data         

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
