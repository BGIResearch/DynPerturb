import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import pickle
from collections import defaultdict


def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_list = []
    #feat_l = []
    node_features_celltype1 = defaultdict(dict) # 存储节点特征，key=(node_id, timestamp)，value=特征
    #idx_list = []

    # 读取数据文件
    with open(data_name) as f:
        next(f)  # 跳过标题行
        for line in f:
            e = line.strip().split(',')
            u = int(float(e[0]))  # 头节点 ID
            i = int(float(e[1]))  # 尾节点 ID
            ts = float(e[2])  # 时间点
            label = int(float(e[3]))  # 标签
            celltype = int(float(e[4]))  # celltype
            
            # 只保留 celltype=1
            if celltype != 0:
                continue

            # 边的属性
            feat_vals = np.array([float(x) for x in e[5:8]])  # celltype, coef_abs, p, -logp
            feat_array = np.array(feat_vals)

            # 节点特征
            head_feat = np.array([float(x) for x in e[8:158]])  # 头节点特征 (1-150)
            tail_feat = np.array([float(x) for x in e[158:]])  # 尾节点特征 (1-150)

            # 记录数据
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            #celltype_list.append(celltype)
            label_list.append(label)
            #idx_list.append(idx + 1)

            feat_list.append(feat_array)

            # # 存储头节点特征
            # if (u, ts, celltype) not in node_features:
            #     node_features[(u, ts, celltype)] = []
            # node_features[(u, ts, celltype)].append(node_feat_u)

            # # 存储尾节点特征
            # if (i, ts, celltype) not in node_features:
            #     node_features[(i, ts, celltype)] = []
            # node_features[(i, ts, celltype)].append(node_feat_i)
            
            # 存储头节点特征到嵌套字典
            node_features_celltype1[u][ts] = head_feat
            # 存储尾节点特征到嵌套字典
            node_features_celltype1[i][ts] = tail_feat

    # 将边特征转换为 NumPy 数组
    feat_array = np.array(feat_list)

    # 对边特征进行 Z-Score 标准化
    mean_vals = feat_array.mean(axis=0)
    std_vals = feat_array.std(axis=0)
    feat_array = (feat_array - mean_vals) / std_vals

    # # 计算节点特征的平均值
    # avg_node_features_celltype1 = {}
    # for (node,ts), feats in node_features_celltype1.items():
    #     avg_node_features_celltype1[(node,ts)] = np.mean(feats, axis=0)  # 平均特征
        
    df = pd.DataFrame({
        'u': u_list,
        'i': i_list,
        'ts': ts_list,
        'celltype': 1,  # 明确标注都为 1
        'label': label_list
    })
    # 直接生成 idx 列
    df['idx'] = range(1, len(df) + 1)  # 生成 idx，从 1 开始
    # 返回数据
    return df, feat_array, node_features_celltype1
def save_dict_as_pickle(node_features_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(node_features_dict, f)


def run(data_name):
    Path("data/").mkdir(parents=True, exist_ok=True)
    PATH = './data/{}.csv'.format(data_name)
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    OUT_FEAT = './data/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './data/ml_{}_node.pkl'.format(data_name)

     # 预处理 (只保留 celltype=1)
    df_celltype1, feat_array, node_features_celltype1 = preprocess(PATH)

    # 保存只包含 celltype=1 的边信息
    df_celltype1.to_csv(OUT_DF, index=False)
    # 保存边特征
    np.save(OUT_FEAT, feat_array)
    # 保存节点特征 (celltype=1)
    save_dict_as_pickle(node_features_celltype1, OUT_NODE_FEAT)

    print(f"数据处理完成，文件已保存：\n数据文件:{OUT_DF}\n边特征文件:{OUT_FEAT}\n节点特征文件:{OUT_NODE_FEAT}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
    parser.add_argument('--data', type=str, help='Dataset name (e.g., wikipedia or reddit)',
                        default='wikipedia')
    args = parser.parse_args()
    run(args.data)

