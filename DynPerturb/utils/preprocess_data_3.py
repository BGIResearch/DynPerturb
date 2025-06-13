import numpy as np
import pandas as pd
from collections import defaultdict


def preprocess(data_name):
    u_list, i_list, ts_list, celltype_list, label_list = [], [], [], [], []
    feat_l = []
    node_features = defaultdict(dict)  # 存储节点特征，key=(node_id, timestamp, celltype)，value=特征
    idx_list = []

    # 读取数据文件
    with open(data_name) as f:
        next(f)  # 跳过标题行
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(float(e[0]))  # 头节点 ID
            i = int(float(e[1]))  # 尾节点 ID
            ts = float(e[2])  # 时间点
            celltype = int(float(e[3]))  # celltype
            label = int(float(e[4]))  # 标签

            # 边的属性
            feat = np.array([float(x) for x in e[4:8]])  # celltype, coef_abs, p, -logp

            # 节点特征
            node_feat_u = np.array([float(x) for x in e[8:158]])  # 头节点特征 (1-150)
            node_feat_i = np.array([float(x) for x in e[158:]])  # 尾节点特征 (1-150)

            # 记录数据
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            celltype_list.append(celltype)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)

            # 存储头节点特征
            if (u, ts, celltype) not in node_features:
                node_features[(u, ts, celltype)] = []
            node_features[(u, ts, celltype)].append(node_feat_u)

            # 存储尾节点特征
            if (i, ts, celltype) not in node_features:
                node_features[(i, ts, celltype)] = []
            node_features[(i, ts, celltype)].append(node_feat_i)

    # 将边特征转换为 NumPy 数组
    feat_l = np.array(feat_l)

    # 对边特征进行 Z-Score 标准化
    mean_vals = feat_l.mean(axis=0)
    std_vals = feat_l.std(axis=0)
    feat_l = (feat_l - mean_vals) / std_vals

    # 计算节点特征的平均值
    avg_node_features = {}
    for key, feats in node_features.items():
        avg_node_features[key] = np.mean(np.array(feats), axis=0)  # 平均特征

    # 返回数据
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'celltype': celltype_list,
                         'label': label_list,
                         'idx': idx_list}), feat_l, avg_node_features


def save_node_features(node_features, output_file):
    """
    保存节点特征到文件中，每个节点的特征以 (node_id, timestamp, celltype) 作为索引。
    """
    with open(output_file, 'w') as f:
        f.write('node_id,timestamp,celltype,' + ','.join([f'feat_{i}' for i in range(150)]) + '\n')
        for (node_id, timestamp, celltype), feature in node_features.items():
            feature_str = ','.join(map(str, feature))
            f.write(f'{node_id},{timestamp},{celltype},{feature_str}\n')


def run(data_name):
    Path("data/").mkdir(parents=True, exist_ok=True)
    PATH = './data/{}.csv'.format(data_name)
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    OUT_FEAT = './data/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './data/ml_{}_node_features.csv'.format(data_name)

    # 预处理数据
    df, feat, node_features = preprocess(PATH)

    # 保存边数据和边特征
    df.to_csv(OUT_DF, index=False)
    np.save(OUT_FEAT, feat)

    # 保存节点特征
    save_node_features(node_features, OUT_NODE_FEAT)

    print(f"数据处理完成，文件已保存：\n数据文件:{OUT_DF}\n边特征文件:{OUT_FEAT}\n节点特征文件:{OUT_NODE_FEAT}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
    parser.add_argument('--data', type=str, help='Dataset name (e.g., wikipedia or reddit)',
                        default='wikipedia')
    args = parser.parse_args()
    run(args.data)

