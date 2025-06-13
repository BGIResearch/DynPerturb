import math
import logging
import time
import sys
import random
import argparse
import pickle
from pathlib import Path
import torch
import numpy as np
from model.tgn_mulit import TGN
from utils.utils_mulit import EarlyStopMonitor, get_neighbor_finder, MLP, RandEdgeSampler
from utils.data_processing_mulit import compute_time_statistics, get_data_node_classification, get_data_mulit,get_data_mulit_0423
from evaluation.evaluation_mulit import eval_node_classification_ddp_new, eval_edge_prediction_add_ddp_new

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

os.environ["SWANLAB_REQUIREMENTS"] = "off"  # 添加此行禁用 pixi 收集
from swanlab.swanlab_settings import Settings
import swanlab
import logging
logging.getLogger("urllib3").setLevel(logging.WARNING)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
settings = Settings(requirements_collect=False, conda_collect=False)
swanlab.login(api_key="tIhGTN0qRK2hMyAyqdl9M")
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

os.chdir("/home/share/huadjyin/home/s_qinhua2/02code/tgn-master/spatiotemporal_mouse/")

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='HumanBone')
parser.add_argument('--bs', type=int, default=64, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true', #default=True,
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                   'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true', default=True,
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=1000, help='Dimensions of the memory for '
                                                                 'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true', default=True,
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true', default=True,
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--use_validation', action='store_true', default=True,
                    help='Whether to use a validation set')
parser.add_argument('--new_node', action='store_true', help='model new node')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
# 其他进程不初始化[6](@ref)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}' + '\
  node-classification.pth'
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}' + '\
  node-classification.pth'

# 改成你有权限的路径
output_dir = "./figs"
os.makedirs(output_dir, exist_ok=True)

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('./log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

num_classes = 3

full_data, node_features, edge_features, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
    get_data_mulit_0423(DATA, use_validation=args.use_validation)

max_idx = max(full_data.unique_nodes)

train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)

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

for i in range(args.n_runs):
    results_path = "./results/{}_node_classification_{}.pkl".format(args.prefix,
                                                                    i) if i > 0 else "./results/{}_node_classification.pkl".format(
        args.prefix)
    Path("./results/").mkdir(parents=True, exist_ok=True)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 初始化进程组
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        swanlab.init(project="TGN_mouse", workspace="zzzzzzzzzzzzz", config=args,settings=settings)  # ← 这行禁用 pixi 自动依赖收集
    tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
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
              num_classes=num_classes)  # 将 num_classes 作为参数传递给 TGN

    # tgn.set_neighbor_finder(train_ngh_finder)
    tgn = tgn.to(device)
    tgn = DDP(tgn, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.debug('Num of training instances: {}'.format(num_instance))
    logger.debug('Num of batches per epoch: {}'.format(num_batch))

    logger.info('TGN models loaded')
    logger.info('Start training node classification task')

    # Setup model, optimizer, etc.
    optimizer = torch.optim.Adam(tgn.parameters(), lr=args.lr)

    val_aucs = []
    train_losses = []

    # best_combined_auc = 0
    best_combined_auc = -float('inf')   # ➜ 保证第一次总能保存
    best_model_epoch = None             # ➜ 训练结束前检查是否赋值
 
    from sklearn.utils.class_weight import compute_class_weight

    weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_data.labels)
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    early_stopper = EarlyStopMonitor(max_round=args.patience)
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_data.sources),
        torch.from_numpy(train_data.destinations),
        torch.from_numpy(train_data.timestamps),
        torch.from_numpy(train_data.edge_idxs),
        torch.from_numpy(train_data.labels)
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    for epoch in range(args.n_epoch):
        logger.info(f"--- Epoch {epoch + 1}/{args.n_epoch} ---")
        start_epoch = time.time()

        # Initialize memory of the model at each epoch
        if USE_MEMORY:
            tgn.module.memory.__init_memory__()
            
        # Train using only training graph
        # tgn.set_neighbor_finder(train_ngh_finder)
        tgn.module.set_neighbor_finder(train_ngh_finder) 
        
        optimizer.zero_grad()

        tgn = tgn.train()

        m_loss = []  

        k = 0
        for batch in train_loader:
            k += 1
            sources_batch, dest_batch, ts_batch, edge_idxs_batch, labels_batch = batch
            # 数据自动分配到当前GPU
            sources_batch = sources_batch.to(device, non_blocking=True)
            destinations_batch = dest_batch.to(device, non_blocking=True)
            timestamps_batch = ts_batch.to(device, non_blocking=True)
            edge_idxs_batch = edge_idxs_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            size = len(sources_batch)

            _, negatives_batch = train_rand_sampler.sample(size)

            optimizer.zero_grad()
            
            # tgn = tgn.train()

            pos_score, neg_score, node_classification_logits = tgn(
                sources_batch, destinations_batch, negatives_batch, timestamps_batch, edge_idxs_batch)

            # 链接预测任务的损失 (使用 BCE 损失)
            pos_labels = torch.ones(len(sources_batch), dtype=torch.float, device=device)
            neg_labels = torch.zeros(len(sources_batch), dtype=torch.float, device=device)
            pos_score = pos_score.squeeze(dim=-1)
            neg_score = neg_score.squeeze(dim=-1)
            link_pred_loss = torch.nn.BCELoss()(pos_score, pos_labels) + torch.nn.BCELoss()(neg_score, neg_labels)

            # 确保 labels_batch 是一个 Tensor 类型
            labels_batch = torch.tensor(labels_batch, dtype=torch.long).to(device)
            # 节点分类任务的损失 (使用 CrossEntropy 损失)
            if node_classification_logits is not None:
                node_classification_loss = torch.nn.CrossEntropyLoss(weight=weights)(node_classification_logits,
                                                                                     labels_batch)
            else:
                node_classification_loss = 0

            # 总损失是两个任务的损失之和
            total_loss = link_pred_loss + node_classification_loss
            total_loss.backward()
            optimizer.step()
            
            # ✅ DDP 汇总损失
            loss_tensor = torch.tensor(total_loss.item(), device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor / dist.get_world_size()

            m_loss.append(avg_loss.item())
            
            if rank == 0:
                swanlab.log({
                    "batch_loss": avg_loss.item(),
                    "epoch": epoch + 1,
                    "batch": k + 1
                })
                    
            
            
            # loss_val = total_loss.item()
            # m_loss.append(loss_val)
            # m_loss.append(total_loss.item())  # 记录每个批次的损失
            # ✅ 每个 batch 的 loss 记录
            # if rank==0:
            #     swanlab.log({
            #         "batch_loss": loss_val,
            #         "epoch": epoch + 1,
            #         "batch": k + 1
            #     })

            if k % 10 == 0 or k == num_batch - 1:
                logger.info(f"[Epoch {epoch + 1}] Batch {k + 1}/{num_batch} - Loss: {total_loss.item():.4f}")
        # train_losses.append(loss / num_batch)
        train_losses.append(np.mean(m_loss))
        if rank==0:
            swanlab.log({"train_loss": train_losses[-1], "epoch": epoch + 1})
        logger.info(f"Epoch {epoch + 1}/{args.n_epoch} - Average Train Loss: {train_losses[-1]:.4f}")

        # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
        # the start of time
        if USE_MEMORY:
            tgn.module.memory.detach_memory()
            
        ### Validation
        # Validation uses the full graph
        tgn.module.set_neighbor_finder(full_ngh_finder)

        if USE_MEMORY:
            # Backup memory at the end of training, so later we can restore it and use it for the
            # validation on unseen nodes
            train_memory_backup = tgn.module.memory.backup_memory()

        # 评估连接预测任务的 AUC
        val_ap_link_prediction, val_auc_link_prediction, val_precision_link_prediction, val_recall_link_prediction, val_f1_link_prediction, val_acc_link_prediction = eval_edge_prediction_add_ddp_new(
            tgn, val_rand_sampler, val_data, NUM_NEIGHBORS, BATCH_SIZE)

        # 评估节点分类任务的 AUC
        val_auc_node_classification, precision, recall, f1, support, pred_prob,true_labels_val = eval_node_classification_ddp_new(
            tgn, val_data, full_data.edge_idxs, BATCH_SIZE, NUM_NEIGHBORS, num_classes)
        
        # print('val_auc_node_classification',val_auc_node_classification)
        # print('val_auc_node_classification',val_auc_node_classification.dtype())

        if USE_MEMORY:
            val_memory_backup = tgn.module.memory.backup_memory()
            # Restore memory we had at the end of training to be used when validating on new nodes.
            # Also backup memory after validation so it can be used for testing (since test edges are
            # strictly later in time than validation edges)
            tgn.module.memory.restore_memory(train_memory_backup)


        # Validate on unseen nodes

        nn_val_ap, nn_val_auc, nn_val_precision, nn_val_recall, nn_val_f1, nn_val_acc = eval_edge_prediction_add_ddp_new(
            model=tgn,
            negative_edge_sampler=val_rand_sampler,
            data=new_node_val_data,
            n_neighbors=NUM_NEIGHBORS)

        if USE_MEMORY:
            # Restore memory we had at the end of validation
            tgn.module.memory.restore_memory(val_memory_backup)

        logger.info(f"[Epoch {epoch + 1}] Val Link Prediction AUC: {val_auc_link_prediction:.4f}")
        for i in range(len(val_auc_node_classification)):
            logger.info(
                f"[Epoch {epoch + 1}] Class {i} - AUC: {val_auc_node_classification[i]:.4f}, Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}, Support: {support[i]:.4f}")

        if rank == 0:
            swanlab.log({
                "val_ap_link_prediction": val_ap_link_prediction,
                "val_auc_link_prediction": val_auc_link_prediction,
                "val_precision_link_prediction": val_precision_link_prediction,
                "val_recall_link_prediction": val_recall_link_prediction,
                "val_f1_link_prediction": val_f1_link_prediction,
                "val_acc_link_prediction": val_acc_link_prediction,
                "nn_val_ap_link_prediction": nn_val_ap,
                "nn_val_auc_link_prediction": nn_val_auc,
                "nn_val_precision_link_prediction": nn_val_precision,
                "nn_val_recall_link_prediction": nn_val_recall,
                "nn_val_f1_link_prediction": nn_val_f1,
                "nn_val_acc_link_prediction": nn_val_acc,
                "val_auc_node_classification": np.mean(val_auc_node_classification),
                "val_precision_node_classification": np.mean(precision),
                "val_recall_node_classification": np.mean(recall),
                "val_f1_node_classification": np.mean(f1),
                "val_support_node_classification": np.mean(support),
                **{f"val_auc_node_classification_{i}": val_auc_node_classification[i] for i in range(num_classes)},
                **{f"val_precision_node_classification_{i}": precision[i] for i in range(num_classes)},
                **{f"val_recall_node_classification_{i}": recall[i] for i in range(num_classes)},
                **{f"val_f1_node_classification_{i}": f1[i] for i in range(num_classes)},
                **{f"val_support_node_classification_{i}": support[i] for i in range(num_classes)},
            })


        # val_auc_node_classification_mean = np.mean(val_auc_node_classification)
        val_f1_node_classification_mean = np.mean(f1)
        # 综合 AUC
        # combined_auc = (val_auc_link_prediction + val_auc_node_classification_mean) / 2  # 或者你可以使用加权平均
        combined_auc = (val_auc_link_prediction + val_f1_node_classification_mean) / 2  # 或者你可以使用加权平均

        # ========= ⏹ DDP 早停判断 =========
        # Step 1: 仅主进程判断是否触发早停
        if dist.get_rank() == 0:
            stop_flag = early_stopper.early_stop_check(combined_auc)
        else:
            stop_flag = False
            
        # Step 2: 同步早停状态到所有进程
        stop_tensor = torch.tensor(int(stop_flag), device=device)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item() == 1:
            logger.info(f"[Rank {dist.get_rank()}] Triggered early stopping at epoch {epoch + 1}")
            break  # 退出训练主循环
        
        
        # ========= 💾 模型保存（仅 rank 0） =========
        if dist.get_rank() == 0 and combined_auc > best_combined_auc:
            best_combined_auc = combined_auc
            best_model_epoch = epoch
            torch.save(tgn.module.state_dict(), get_checkpoint_path(epoch))
            logger.info(f"✅ Best model saved at epoch {epoch + 1} with Combined AUC: {best_combined_auc:.4f}")
            logger.info(f"   ↳ Link Prediction AUC: {val_auc_link_prediction:.4f}")
            # logger.info(f"   ↳ Node Classification Auc (Mean): {val_auc_node_classification_mean:.4f}")
            logger.info(f"   ↳ Node Classification F1 (Mean): {val_f1_node_classification_mean:.4f}")
            
        # ========= 📊 swanlab 记录 =========
        if dist.get_rank() == 0:
            swanlab.log({
                "combined_auc": combined_auc,
                "early_stop_rounds": early_stopper.num_round,
                "best_auc_so_far": best_combined_auc,
                "epoch": epoch + 1
            })
        
        # if best_model_epoch is None:
        #     raise RuntimeError("❌ No best model was saved. Check if training diverged or AUC is invalid.")



    # 加载最佳模型
    # logger.info(f'Loading the best model from epoch {best_model_epoch}')
    # tgn.module.load_state_dict(torch.load(get_checkpoint_path(best_model_epoch)))
    
    if dist.get_rank() == 0:
        torch.save(tgn.module.state_dict(), get_checkpoint_path(epoch))
        logger.info(f"... saved ...")

    # 所有进程同步，确保文件写入完成再读
    dist.barrier()
    

    # ========= ✅ 同步 best_model_epoch 到所有进程 =========
    best_model_epoch_tensor = torch.tensor(
        best_model_epoch if dist.get_rank() == 0 else -1,
        device=device
    )
    dist.broadcast(best_model_epoch_tensor, src=0)
    best_model_epoch = best_model_epoch_tensor.item()
    if best_model_epoch == -1:
        raise RuntimeError("❌ No best model was saved. Check if training diverged or AUC is invalid.")


    # 所有进程一起加载
    tgn.module.load_state_dict(torch.load(get_checkpoint_path(best_model_epoch)))


    # Training has finished, we have loaded the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen
    # nodes
    if USE_MEMORY:
        val_memory_backup = tgn.module.memory.backup_memory()

 
      ### Test
    tgn.module.embedding_module.neighbor_finder = full_ngh_finder

    # 测试集评估
    test_ap_link_prediction, test_auc_link_prediction, test_precision_link_prediction, test_recall_link_prediction, test_f1_link_prediction, test_acc_link_prediction = eval_edge_prediction_add_ddp_new(
        tgn, test_rand_sampler, test_data, NUM_NEIGHBORS, BATCH_SIZE)

    test_auc_node_classification, precision, recall, f1, support, pred_prob, true_labels= eval_node_classification_ddp_new(
        tgn, test_data, full_data.edge_idxs, BATCH_SIZE, NUM_NEIGHBORS, num_classes)

    if USE_MEMORY:
        tgn.module.memory.restore_memory(val_memory_backup)

    nn_test_ap, nn_test_auc, nn_test_precision, nn_test_recall, nn_test_f1, nn_test_acc = eval_edge_prediction_add_ddp_new(
        model=tgn,
        negative_edge_sampler=nn_test_rand_sampler,
        data=new_node_test_data,
        n_neighbors=NUM_NEIGHBORS)

    # Log to swanlab
    if rank == 0:
        swanlab.log({
            "test_ap_link_prediction": test_ap_link_prediction,
            "test_auc_link_prediction": test_auc_link_prediction,
            "test_precision_link_prediction": test_precision_link_prediction,
            "test_recall_link_prediction": test_recall_link_prediction,
            "test_f1_link_prediction": test_f1_link_prediction,
            "test_acc_link_prediction": test_acc_link_prediction,
            "nn_test_ap_link_prediction": nn_test_ap,
            "nn_test_auc_link_prediction": nn_test_auc,
            "nn_test_precision_link_prediction": nn_test_precision,
            "nn_test_recall_link_prediction": nn_test_recall,
            "nn_test_f1_link_prediction": nn_test_f1,
            "nn_test_acc_link_prediction": nn_test_acc,
            "test_auc_node_classification": np.mean(test_auc_node_classification),
            "test_precision_node_classification": np.mean(precision),
            "test_recall_node_classification": np.mean(recall),
            "test_f1_node_classification": np.mean(f1),
            "test_support_node_classification": np.mean(support),
            **{f"test_auc_node_classification_{i}": test_auc_node_classification[i] for i in range(num_classes)},
            **{f"test_precision_node_classification_{i}": precision[i] for i in range(num_classes)},
            **{f"test_recall_node_classification_{i}": recall[i] for i in range(num_classes)},
            **{f"test_f1_node_classification_{i}": f1[i] for i in range(num_classes)},
            **{f"test_support_node_classification_{i}": support[i] for i in range(num_classes)},
        })
        # === 🎯 可视化混淆矩阵 + AUC 柱状图 ===
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        y_true = true_labels
        y_pred = np.argmax(pred_prob, axis=1)
        class_names = [f"Class {i}" for i in range(num_classes)]

        # 原始混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{args.prefix}-{args.data}_confusion_matrix.pdf"))
        plt.close()

        # 归一化混淆矩阵
        cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Normalized)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{args.prefix}-{args.data}_confusion_matrix_normalized.pdf"))
        plt.close()

        # AUC 柱状图
        plt.figure(figsize=(10, 5))
        plt.bar(class_names, test_auc_node_classification)
        plt.ylabel('AUC Score')
        plt.title('Per-Class AUC')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{args.prefix}-{args.data}_auc_bar_chart.pdf"))
        plt.close()
    logger.info(f'Test Link Prediction AUC: {test_auc_link_prediction}')
    logger.info(f'Test Node Classification AUC: {test_auc_node_classification}')
    logger.info('Saving TGN model')

    if USE_MEMORY:
        # Restore memory at the end of validation (save a model which is ready for testing)
        tgn.module.memory.restore_memory(val_memory_backup)
    torch.save(tgn.module.state_dict(), MODEL_SAVE_PATH)
    logger.info('TGN model saved')

# Finish the swanlab run
swanlab.finish()
dist.destroy_process_group()

