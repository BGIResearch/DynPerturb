import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    node_feat_l={}
    idx_list = []

    # 读取数据文件
    with open(data_name) as f:
        next(f)  # 跳过标题行
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(float(e[0]))
            i = int(float(e[1]))
            ts = float(e[2]) # 时间戳改为整数类型
            label = int(float(e[3]))  # 标签改为整数类型

            feat = np.array([float(x) for x in e[4:8]])
            #feat = np.array([float(x) for x in e[4:]])
            node_feat_u = np.array([float(x) for x in e[8:158]])  # 头节点特征 (1-150)
            node_feat_i = np.array([float(x) for x in e[158:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
            
            # 存储头节点和尾节点特征
            if u not in node_feat_l:
                node_feat_l[u] = []
            node_feat_l[u].append(node_feat_u)

            if i not in node_feat_l:
                node_feat_l[i] = []
            node_feat_l[i].append(node_feat_i)
    
    # 将特征列表转换为 NumPy 数组
    feat_l = np.array(feat_l)
    # Z-Score 标准化
    mean_vals = feat_l.mean(axis=0)
    std_vals = feat_l.std(axis=0)
    feat_l = (feat_l - mean_vals) / std_vals
    
    # 计算每个节点的特征的平均值
    avg_node_feats = {}
    for node, feats in node_feat_l.items():
        avg_node_feats[node] = np.mean(np.array(feats), axis=0)  # 平均特征
    
    # 为每个边赋予头节点和尾节点的平均特征
    node_feat_l = np.zeros((len(u_list), 150))  # 假设每个节点的特征维度是150
    for idx, u in enumerate(u_list):
        node_feat_l[idx] = avg_node_feats.get(u, np.zeros(150))  # 头节点特征
    
    tail_node_feat_l = np.zeros((len(i_list), 150))  # 尾节点特征存储
    for idx, i in enumerate(i_list):
        tail_node_feat_l[idx] = avg_node_feats.get(i, np.zeros(150))  # 尾节点特征
    
    # 返回 DataFrame 和特征数组
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), feat_l, node_feat_l


def reindex(df):
    new_df = df.copy()
    # 直接对 u 和 i 进行重新索引，无需区分是否是二部图
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
    return new_df


def run(data_name):
    Path("data/").mkdir(parents=True, exist_ok=True)
    PATH = './data/{}.csv'.format(data_name)
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    OUT_FEAT = './data/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

    # 预处理数据
    df, feat, node_feat = preprocess(PATH)
    new_df = reindex(df)

    # 保存处理后的数据
    new_df.to_csv(OUT_DF, index=False)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, node_feat)

    print(f"数据处理完成，文件已保存：\n数据文件:{OUT_DF}\n特征文件:{OUT_FEAT}\n节点特征文件:{OUT_NODE_FEAT}")


# 解析命令行参数
parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')

args = parser.parse_args()

# 运行数据预处理
run(args.data)
