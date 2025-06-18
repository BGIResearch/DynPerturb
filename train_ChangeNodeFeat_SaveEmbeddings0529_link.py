
from evaluation.evaluation import (
    eval_edge_prediction_add,
    eval_edge_prediction_add_ddp_new1,
    eval_node_classification_ddp_new1,
    eval_node_classification_ddp_label
)


from model.NetModel import NetModel

from modules.MemoryModule import Memory, GRUMemoryUpdater, get_memory_updater

from modules.MessageOps import get_message_function, get_message_aggregator

from utils.DataLoader import (
    get_data_mulit_0423,
    get_data_mulit_0512,
    get_data_node_classification
)

from utils.utils import (
    MLP,
    MergeLayer,
    FocalLoss,
    EarlyStopMonitor,
    RandEdgeSampler,
    get_neighbor_finder,
    NeighborFinder
)





import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import shutil  # 新增用于复制文件
import os

from copy import deepcopy
import json
from collections import defaultdict
# from evaluation.evaluation import eval_edge_prediction
# from model.netmodel_0208 import NetModel
# from model.netmodel_0417 import NetModel
# from model.netmodel_0208 import NetModel
from model.netmodel_mulit0606 import NetModel
# from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
# from utils.data_processing0208 import get_data,get_data_0313,get_data_0422, get_data_0422_70_15_15,get_data_0423_60_15_25,get_data_0423_70_15_15,compute_time_statistics

# 修改导入，使用修改后的 embedding_module_saved
# from embedding_module_saved import get_embedding_module  # 修改导入路径

os.environ["SWANLAB_REQUIREMENTS"] = "off"  # 添加此行禁用 pixi 收集
from swanlab.swanlab_settings import Settings
import swanlab
import logging
logging.getLogger("urllib3").setLevel(logging.WARNING)



os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

os.chdir("/home/share/huadjyin/home/s_qinhua2/02code/netmodel-master/kidney/") 
torch.manual_seed(0)
np.random.seed(0)
settings = Settings(requirements_collect=False, conda_collect=False)
swanlab.login(api_key="tIhGTN0qRK2hMyAyqdl9M")
### Argument and global variables
parser = argparse.ArgumentParser('NetModel self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='celltype32')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',# default=True,
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn","lstm"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
# parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
#                                                                 'each user')
parser.add_argument('--memory_dim', type=int, default=1000, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--num_classes', type=int, default=4, help='class num')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--use_validation', action='store_true', default=True,
                    help='Whether to use a validation set')
# parser.add_argument('--dyrep', action='store_true',
#                     help='Whether to run the dyrep model')

# parser.add_argument('--change_only', action='store_true', default=True, 
#                     help='Skip training and only run testing by default')
# parser.add_argument('--node_id_modify', type=int, default=103, help='Node ID to modify (default is 0)')  # 新增节点ID的命令行输入
# parser.add_argument('--time_modify', type=int, default=43, help='Time to modify (default is 0)')  # 新增节点ID的命令行输入



try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)
  
swanlab.init(project="NetModel_KO", workspace="zzzzzzzzzzzzz", config=args,settings=settings)  # ← 这行禁用 pixi 自动依赖收集


BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
NumClasses=args.num_classes
# node_to_modify=args.node_id_modify
# modify_time=args.time_modify

# **所有输出文件夹都基于 DATA 名称**
MODEL_DIR = f'./saved_models/{DATA}/'
CHECKPOINT_DIR = f'./saved_checkpoints/{DATA}/'
EMBEDDING_DIR = f'./saved_embeddings/{DATA}/'
RESULTS_DIR = f'./results/{DATA}/'
LOG_DIR = f'./log/{DATA}/'

# **创建文件夹**
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
Path(EMBEDDING_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = f'{MODEL_DIR}/best_model.pth'
get_checkpoint_path = lambda epoch: f'{CHECKPOINT_DIR}/checkpoint_epoch_{epoch}.pth'
RESULTS_PATH = f"{RESULTS_DIR}/results.pkl"
BEST_EMBEDDING_PATH = f"{EMBEDDING_DIR}/embeddings_best.json"

# Path("./saved_models/").mkdir(parents=True, exist_ok=True)
# Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
# MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
# get_checkpoint_path = lambda \
#     epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

# ### 设置嵌入保存目录
# Path("./saved_embeddings/").mkdir(parents=True, exist_ok=True)
# EMBEDDING_SAVE_DIR = './saved_embeddings/'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# Path("log/").mkdir(parents=True, exist_ok=True)
# fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh = logging.FileHandler(f'{LOG_DIR}/train.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

# node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_and_change_data(
#     DATA,
#     node_to_modify,  # 需要修改的节点ID
#     modify_time  # 需要修改的时间戳
# )
# full_data, node_features, edge_features, train_data, val_data, test_data, new_node_val_data, new_node_test_data= \
#             get_data_mulit_0423(DATA, use_validation=args.use_validation)
full_data, node_features, edge_features, train_data, val_data, test_data, new_node_val_data, new_node_test_data= \
            get_data_mulit_0512(DATA, use_validation=args.use_validation)
# node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
# new_node_test_data = get_data_0423_70_15_15(DATA,
#                               different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)

# full_data, node_features, edge_features, train_data, val_data, test_data, new_node_val_data, new_node_test_data= \
#             get_data_mulit(DATA, use_validation=args.use_validation)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)



logger.info("🔹loading the best model for infering...")

# **加载已经训练好的模型**
# MODEL_SAVE_PATH = f'./saved_models/{DATA}/best_model.pth'
# MODEL_SAVE_PATH = f'/home/share/huadjyin/home/s_qinhua2/02code/netmodel-master/kidney/saved_models/aPT-B/aPT-B_lr_3e-4_best_model.pth'
MODEL_SAVE_PATH = f'/home/share/huadjyin/home/s_qinhua2/02code/netmodel-master/kidney/saved_models/PT-S2-D/PT-S2-D_only_link_best_model.pth'
if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"❌ Best model not found at {MODEL_SAVE_PATH}, please train first.")

# **初始化 NetModel 模型**
netmodel = NetModel(neighbor_finder=train_ngh_finder, node_features=node_features,
              edge_features=edge_features, device=device,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
              message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
              memory_update_at_start=not args.memory_update_at_end,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator,
              memory_updater_type=args.memory_updater,
              n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message,
              num_classes=NumClasses)

netmodel = netmodel.to(device)
netmodel.load_state_dict(torch.load(MODEL_SAVE_PATH), strict=False)
# netmodel.load_state_dict(torch.load(MODEL_SAVE_PATH), strict=True)# 加载权重
netmodel.eval()
logger.info(f'✅ Successfully loaded the best model from {MODEL_SAVE_PATH}')


logger.info("🔹 Starting batched inference with memory propagation (no per-time grouping)...")

# ✅ 初始化 memory
if USE_MEMORY:
    netmodel.memory.__init_memory__()
netmodel.set_neighbor_finder(full_ngh_finder)

saved_embeddings = defaultdict(list)

# ✅ 准备全图有序事件（已升序排列）
sources = full_data.sources
destinations = full_data.destinations
timestamps = full_data.timestamps
edge_idxs = full_data.edge_idxs

num_events = len(sources)

# ✅ 分批处理
with torch.no_grad():
    for start in range(0, num_events, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_events)

        src = torch.from_numpy(sources[start:end]).to(device)
        dst = torch.from_numpy(destinations[start:end]).to(device)
        ts = torch.from_numpy(timestamps[start:end]).to(device)
        eid = torch.from_numpy(edge_idxs[start:end]).to(device)
        neg = dst.clone()

        emb_src, emb_dst, _ = netmodel.compute_temporal_embeddings(src, dst, neg, ts, eid)

        for i in range(len(src)):
            s_id = int(src[i])
            d_id = int(dst[i])
            t_val = float(ts[i])
            saved_embeddings[s_id].append({
                "timestamp": t_val,
                "embedding": emb_src[i].detach().cpu().numpy().tolist()
            })
            saved_embeddings[d_id].append({
                "timestamp": t_val,
                "embedding": emb_dst[i].detach().cpu().numpy().tolist()
            })

# ✅ 保存 JSON
Path(EMBEDDING_DIR).mkdir(parents=True, exist_ok=True)
save_path = os.path.join(EMBEDDING_DIR, f"embeddings_{args.prefix}.json")
with open(save_path, 'w') as f:
    json.dump(saved_embeddings, f)

logger.info(f"✅ Embeddings with memory propagation saved to {save_path}")
# ✅ 日志：总共保存了多少个嵌入点
total_embs = sum(len(v) for v in saved_embeddings.values())
logger.info(f"📦 Total node-time embeddings saved: {total_embs}")
logger.info(f"📁 Saved JSON file to {save_path}")
# Finish the swanlab run
swanlab.finish()

