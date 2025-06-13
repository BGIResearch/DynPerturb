# import numpy as np
# import pandas as pd
# from pathlib import Path
# import argparse


# def preprocess(data_name, batch_size=100000):
#     """
#     预处理数据文件，逐批读取数据，返回生成器，每次处理指定的批量大小。
#     """
#     u_list, i_list, ts_list, label_list = [], [], [], []
#     feat_l = []
#     idx_list = []

#     with open(data_name) as f:
#         next(f)  # 跳过标题行
#         for idx, line in enumerate(f):
#             e = line.strip().split(',')
#             # 检查是否为空字符串，若为空则跳过这行
#             if e[0] == '' or e[1] == '':
#                 continue
#             u = int(float(e[0]))
#             i = int(float(e[1]))
#             ts = int(float(e[2]))  # 时间戳改为整数类型
#             label = int(float(e[3]))  # 标签改为整数类型

#             feat = np.array([float(x) if x else 0.0 for x in e[4:]])  # 如果空值，设为 0.0

#             u_list.append(u)
#             i_list.append(i)
#             ts_list.append(ts)
#             label_list.append(label)
#             idx_list.append(idx)
#             feat_l.append(feat)

#             if len(u_list) == batch_size:
#                 yield pd.DataFrame({'u': u_list, 'i': i_list, 'ts': ts_list, 'label': label_list, 'idx': idx_list}), np.array(feat_l)
#                 u_list, i_list, ts_list, label_list, idx_list, feat_l = [], [], [], [], [], []  # 清空以继续处理下一批

#         # 处理剩下的未达到批量大小的数据
#         if u_list:
#             yield pd.DataFrame({'u': u_list, 'i': i_list, 'ts': ts_list, 'label': label_list, 'idx': idx_list}), np.array(feat_l)


# def reindex(df):
#     """
#     重新索引数据,如果不是二部图(即gene可以同时出现在source和target中),则直接对所有节点统一编号。
#     """
#     new_df = df.copy()

#     # 对u列和i列分别加1
#     new_df.u += 1
#     new_df.i += 1
#     new_df.idx += 1

#     return new_df


# def run(data_name, batch_size=100000):
#     """
#     主运行函数，分批读取数据，预处理并保存结果。
#     """
#     Path("data/").mkdir(parents=True, exist_ok=True)
#     PATH = './data/{}.csv'.format(data_name)
#     OUT_DF = './data/ml_{}.csv'.format(data_name)
#     OUT_FEAT = './data/ml_{}.npy'.format(data_name)  # 最终合并后的特征文件
#     OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

#     # 初始化一个大文件，逐批写入数据
#     first_batch = True
#     feat_shape = None
#     batch_num = 0

#     # 临时特征文件列表，用于逐步合并
#     temp_feat_files = []

#     # 逐批处理数据
#     for df_batch, feat_batch in preprocess(PATH, batch_size=batch_size):
#         new_df_batch = reindex(df_batch)

#         # 将批次数据追加到输出文件中
#         mode = 'w' if first_batch else 'a'
#         header = first_batch  # 只有第一次写入时保留表头
#         new_df_batch.to_csv(OUT_DF, mode=mode, header=header, index=False)

#         # 保存每个批次的特征到单独的文件
#         temp_feat_file = f'./data/ml_{data_name}_feat_batch_{batch_num}.npy'
#         np.save(temp_feat_file, feat_batch)
#         temp_feat_files.append(temp_feat_file)
#         batch_num += 1

#         # 初始化空特征占位符
#         if first_batch:
#             feat_shape = feat_batch.shape[1]
#             first_batch = False

#     # 随机生成节点特征，初始化为零数组，维度为 (节点数量, 172)
#     max_idx = max(new_df_batch.u.max(), new_df_batch.i.max())
#     rand_feat = np.zeros((max_idx + 1, 172))
#     np.save(OUT_NODE_FEAT, rand_feat)

#     # 最后将所有临时特征文件合并为一个大特征文件
#     with open(OUT_FEAT, 'wb') as f_out:
#         for temp_file in temp_feat_files:
#             batch_feat = np.load(temp_file)
#             np.save(f_out, batch_feat)

#     print(f"数据处理完成，文件已保存：\n数据文件:{OUT_DF}\n特征文件:{OUT_FEAT}\n节点特征文件:{OUT_NODE_FEAT}")


# # 解析命令行参数
# parser = argparse.ArgumentParser('TGN数据预处理接口')
# parser.add_argument('--data', type=str, help='数据集名称 (例如:wikipedia或reddit)', default='wikipedia')
# parser.add_argument('--batch_size', type=int, default=100000, help='批次大小')

# # 获取命令行参数
# args = parser.parse_args()

# # 运行数据预处理
# run(args.data, batch_size=args.batch_size)


# import numpy as np
# import pandas as pd
# from pathlib import Path
# import argparse


# def preprocess(data_name, batch_size=100000):
#     """
#     预处理数据文件，逐批读取数据，返回生成器，每次处理指定的批量大小。
#     """
#     u_list, i_list, ts_list, label_list = [], [], [], []
#     feat_l = []
#     idx_list = []
#     global_idx = 0  # 使用全局 idx 计数器

#     with open(data_name) as f:
#         next(f)  # 跳过标题行
#         for line in f:
#             e = line.strip().split(',')
#             # 检查是否为空字符串，若为空则跳过这行
#             if e[0] == '' or e[1] == '':
#                 continue
#             u = int(float(e[0]))
#             i = int(float(e[1]))
#             ts = int(float(e[2]))  # 时间戳改为整数类型
#             label = int(float(e[3]))  # 标签改为整数类型

#             feat = np.array([float(x) if x else 0.0 for x in e[4:]])  # 如果空值，设为 0.0

#             u_list.append(u)
#             i_list.append(i)
#             ts_list.append(ts)
#             label_list.append(label)
#             idx_list.append(global_idx)  # 使用全局索引
#             global_idx += 1
#             feat_l.append(feat)

#             if len(u_list) == batch_size:
#                 yield pd.DataFrame({'u': u_list, 'i': i_list, 'ts': ts_list, 'label': label_list, 'idx': idx_list}), np.array(feat_l)
#                 u_list, i_list, ts_list, label_list, idx_list, feat_l = [], [], [], [], [], []  # 清空以继续处理下一批

#         # 处理剩下的未达到批量大小的数据
#         if u_list:
#             yield pd.DataFrame({'u': u_list, 'i': i_list, 'ts': ts_list, 'label': label_list, 'idx': idx_list}), np.array(feat_l)


# def reindex(df):
#     """
#     重新索引数据,如果不是二部图(即gene可以同时出现在source和target中),则直接对所有节点统一编号。
#     """
#     new_df = df.copy()

#     # 对u列和i列分别加1
#     new_df.u += 1
#     new_df.i += 1

#     return new_df


# def run(data_name, batch_size=100000):
#     """
#     主运行函数，分批读取数据，预处理并保存结果。
#     """
#     Path("data/").mkdir(parents=True, exist_ok=True)
#     PATH = './data/{}.csv'.format(data_name)
#     OUT_DF = './data/ml_{}.csv'.format(data_name)
#     OUT_FEAT = './data/ml_{}.npy'.format(data_name)  # 最终合并后的特征文件
#     OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

#     # 初始化一个大文件，逐批写入数据
#     first_batch = True
#     feat_shape = None
#     batch_num = 0

#     # 临时特征文件列表，用于逐步合并
#     temp_feat_files = []

#     # 逐批处理数据
#     for df_batch, feat_batch in preprocess(PATH, batch_size=batch_size):
#         new_df_batch = reindex(df_batch)

#         # 将批次数据追加到输出文件中
#         mode = 'w' if first_batch else 'a'
#         header = first_batch  # 只有第一次写入时保留表头
#         new_df_batch.to_csv(OUT_DF, mode=mode, header=header, index=False)

#         # 保存每个批次的特征到单独的文件
#         temp_feat_file = f'./data/ml_{data_name}_feat_batch_{batch_num}.npy'
#         np.save(temp_feat_file, feat_batch)
#         temp_feat_files.append(temp_feat_file)
#         batch_num += 1

#         # 初始化空特征占位符
#         if first_batch:
#             feat_shape = feat_batch.shape[1]
#             first_batch = False

#     # 随机生成节点特征，初始化为零数组，维度为 (节点数量, 172)
#     max_idx = max(new_df_batch.u.max(), new_df_batch.i.max())
#     rand_feat = np.zeros((max_idx + 1, 172))
#     np.save(OUT_NODE_FEAT, rand_feat)

#     # 最后将所有临时特征文件合并为一个大特征文件
#     with open(OUT_FEAT, 'wb') as f_out:
#         first_batch = True
#         for temp_file in temp_feat_files:
#             batch_feat = np.load(temp_file)
#             if first_batch:
#                 np.save(f_out, batch_feat)  # 保存第一批数据
#                 first_batch = False
#             else:
#                 # 追加后续的批次特征到同一个文件
#                 with open(OUT_FEAT, 'ab') as f_append:
#                     np.save(f_append, batch_feat)

#     print(f"数据处理完成，文件已保存：\n数据文件:{OUT_DF}\n特征文件:{OUT_FEAT}\n节点特征文件:{OUT_NODE_FEAT}")


# # 解析命令行参数
# parser = argparse.ArgumentParser('TGN数据预处理接口')
# parser.add_argument('--data', type=str, help='数据集名称 (例如:wikipedia或reddit)', default='wikipedia')
# parser.add_argument('--batch_size', type=int, default=100000, help='批次大小')

# # 获取命令行参数
# args = parser.parse_args()

# # 运行数据预处理
# run(args.data, batch_size=args.batch_size)

import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    node_feat_l=[]
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

            feat = np.array([float(x) for x in e[4:7]])
            #feat = np.array([float(x) for x in e[4:]])
            node_feat= np.array([float(x) for x in e[8:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
            
            node_feat_l.append(node_feat)
    
    # 将特征列表转换为 NumPy 数组
    feat_l = np.array(feat_l)
    # Z-Score 标准化
    mean_vals = feat_l.mean(axis=0)
    std_vals = feat_l.std(axis=0)
    feat_l = (feat_l - mean_vals) / std_vals
    
    # 将特征列表转换为 NumPy 数组
    node_feat_l = np.array(node_feat_l)
    # Z-Score 标准化
    node_mean_vals = node_feat_l.mean(axis=0)
    node_std_vals = node_feat_l.std(axis=0)
    node_feat_l = (node_feat_l - node_mean_vals) / node_std_vals
    
    # 返回 DataFrame 和特征数组
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l), np.array(node_feat_l)


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

    # 添加空特征占位符
    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])
    
    empty = np.zeros(node_feat.shape[1])[np.newaxis, :]
    node_feat = np.vstack([empty, node_feat])

    # 随机生成节点特征，初始化为零数组
    #max_idx = max(new_df.u.max(), new_df.i.max())
    #rand_feat = np.zeros((max_idx + 1, 172))

    # 保存处理后的数据
    new_df.to_csv(OUT_DF, index=False)
    np.save(OUT_FEAT, feat)
    #np.save(OUT_NODE_FEAT, rand_feat)
    np.save(OUT_NODE_FEAT, node_feat)

    print(f"数据处理完成，文件已保存：\n数据文件:{OUT_DF}\n特征文件:{OUT_FEAT}\n节点特征文件:{OUT_NODE_FEAT}")


# 解析命令行参数
parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')

args = parser.parse_args()

# 运行数据预处理
run(args.data)
