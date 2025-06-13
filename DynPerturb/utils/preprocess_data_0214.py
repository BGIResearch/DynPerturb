import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import pickle
from collections import defaultdict

def preprocess(data_name, celltype_filter=None):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_list = []
    node_features = defaultdict(dict)  # 存储节点特征，key=(node_id, timestamp)，value=特征

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
            
            # 只保留指定的 celltype（如果 celltype_filter 为 None，则保留所有）
            if celltype_filter is not None and celltype != celltype_filter:
                continue

            # 边的属性
            feat_vals = np.array([float(x) for x in e[5:8]])  # coef_abs, p, -logp
            feat_array = np.array(feat_vals)

            # 节点特征
            head_feat = np.array([float(x) for x in e[8:158]])  # 头节点特征 (1-150)
            tail_feat = np.array([float(x) for x in e[158:]])  # 尾节点特征 (1-150)

            # 记录数据
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            feat_list.append(feat_array)

            # 存储头节点特征到嵌套字典
            node_features[u][ts] = head_feat
            # 存储尾节点特征到嵌套字典
            node_features[i][ts] = tail_feat

    # 将边特征转换为 NumPy 数组
    feat_array = np.array(feat_list)

    # 对边特征进行 Z-Score 标准化
    mean_vals = feat_array.mean(axis=0)
    std_vals = feat_array.std(axis=0)
    feat_array = (feat_array - mean_vals) / std_vals

    # 生成 DataFrame
    df = pd.DataFrame({
        'u': u_list,
        'i': i_list,
        'ts': ts_list,
        'celltype': [celltype_filter] * len(u_list),
        'label': label_list
    })
    
    # 生成 idx 列
    df['idx'] = range(1, len(df) + 1)  # 生成 idx，从 1 开始
    
    return df, feat_array, node_features

def save_dict_as_pickle(node_features_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(node_features_dict, f)

def run(data_name):
    Path("data/all_data").mkdir(parents=True, exist_ok=True)
    PATH = './data/all_data/{}.csv'.format(data_name)
    
    # 预处理所有 celltypes
    celltypes = set()  # 用于保存所有的 celltype
    with open(PATH) as f:
        next(f)
        for line in f:
            e = line.strip().split(',')
            celltypes.add(int(float(e[4])))  #取 celltype 列的唯一值
    
    # 处理每个 celltype
    for celltype in celltypes:
        print(f"Processing celltype {celltype}...")

        # 文件名包含 celltype 信息
        OUT_DF = f'./data/all_data/ml_{data_name}_celltype{celltype}.csv'
        OUT_FEAT = f'./data/all_data/ml_{data_name}_celltype{celltype}.npy'
        OUT_NODE_FEAT = f'./data/all_data/ml_{data_name}_celltype{celltype}_node.pkl'

        # 预处理当前 celltype
        df_celltype, feat_array, node_features_celltype = preprocess(PATH, celltype_filter=celltype)

        # 保存只包含当前 celltype 的边信息
        df_celltype.to_csv(OUT_DF, index=False)
        # 保存边特征
        np.save(OUT_FEAT, feat_array)
        # 保存节点特征
        save_dict_as_pickle(node_features_celltype, OUT_NODE_FEAT)

        print(f"数据处理完成，文件已保存：\n数据文件:{OUT_DF}\n边特征文件:{OUT_FEAT}\n节点特征文件:{OUT_NODE_FEAT}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
    parser.add_argument('--data', type=str, help='Dataset name (e.g., wikipedia or reddit)', default='wikipedia')
    args = parser.parse_args()
    run(args.data)
