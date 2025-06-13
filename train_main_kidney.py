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

from evaluation.evaluation import eval_edge_prediction,eval_edge_prediction_add_1
# from model.tgn_0417 import TGN
from model.tgn_0208 import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing0208 import get_data,get_data_0313,get_data_0422, get_data_0422_70_15_15,get_data_0423_60_15_25,get_data_0423_70_15_15,compute_time_statistics

# os.environ["WANDB_API_KEY"] = "6d7e1c88be5af66eb5de8e16f39751a37f4110a0"

# import wandb
# wandb.login()

# 修改导入，使用修改后的 embedding_module_saved
# from embedding_module_saved import get_embedding_module  # 修改导入路径

# import wandb
os.environ["SWANLAB_REQUIREMENTS"] = "off"  # 添加此行禁用 pixi 收集
from swanlab.swanlab_settings import Settings
import swanlab
import logging
logging.getLogger("urllib3").setLevel(logging.WARNING)

os.chdir("/home/share/huadjyin/home/s_qinhua2/02code/tgn-master/kidney/") 
torch.manual_seed(0)
np.random.seed(0)
# wandb.login(key='6d7e1c88be5af66eb5de8e16f39751a37f4110a0')
settings = Settings(requirements_collect=False, conda_collect=False)
swanlab.login(api_key="tIhGTN0qRK2hMyAyqdl9M")
# wandb.login()
### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='celltype32')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')#200
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='Number of neighbors to sample')#10
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')#0.1
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
# parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
#                                                                 'each user')
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
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)
  
# wandb.init(project='TGN_kidney', config=args)
swanlab.init(project="TGN_kidney", workspace="zzzzzzzzzzzzz", config=args,settings=settings)  # ← 这行禁用 pixi 自动依赖收集
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

MODEL_SAVE_PATH = f'{MODEL_DIR}/{args.prefix}_best_model.pth'
get_checkpoint_path = lambda epoch: f'{CHECKPOINT_DIR}/{args.prefix}_checkpoint_epoch_{epoch}.pth'
#RESULTS_PATH = f"{RESULTS_DIR}/results.pkl"
BEST_EMBEDDING_PATH = f"{EMBEDDING_DIR}/{args.prefix}_embeddings_best.json"

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

### Extract data for training, validation and testing
# node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
# new_node_test_data = get_data_0423_60_15_25(DATA,
#                               different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data_0423_70_15_15(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)

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

for i in range(args.n_runs):
  results_path = f"{RESULTS_DIR}/{DATA}_{i}.pkl" if i > 0 else f"{RESULTS_DIR}/{args.prefix}_{DATA}.pkl"
  Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# for i in range(args.n_runs):
#   results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
#   Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
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
            dyrep=args.dyrep)
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)
  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  best_loss = 0  # 假设 val_ap 越高越好
  best_epoch = -1  # 初始化最佳 epoch
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    logger.info('start {} epoch'.format(epoch))
    for k in range(0, num_batch, args.backprop_every):
      loss = 0
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]

        size = len(sources_batch)
        #_, negatives_batch = train_rand_sampler.sample(size*2)
        _, negatives_batch = train_rand_sampler.sample(size)

        with torch.no_grad():
          pos_label = torch.ones(size, dtype=torch.float, device=device)
          neg_label = torch.zeros(size, dtype=torch.float, device=device)

        tgn = tgn.train()

        pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                                                            timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
        

        loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

      loss /= args.backprop_every

      loss.backward()
      optimizer.step()
      # m_loss.append(loss.item())
      step_loss = loss.item()
      m_loss.append(step_loss)
      swanlab.log({'train_loss_step': step_loss})

      # Log training loss to wandb
      # wandb.log({'train_loss': np.mean(m_loss), 'epoch': epoch})
      swanlab.log({'train_loss': np.mean(m_loss), 'epoch': epoch})
      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
      if USE_MEMORY:
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation
    # Validation uses the full graph
    tgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
      train_memory_backup = tgn.memory.backup_memory()

    # val_ap, val_auc = eval_edge_prediction(model=tgn,
    #                                                         negative_edge_sampler=val_rand_sampler,
    #                                                         data=val_data,
    #                                                         n_neighbors=NUM_NEIGHBORS)
    val_ap, val_auc,val_acc, val_f1 = eval_edge_prediction_add_1(model=tgn,
                                                            negative_edge_sampler=val_rand_sampler,
                                                            data=val_data,
                                                            n_neighbors=NUM_NEIGHBORS)
    if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()
      # Restore memory we had at the end of training to be used when validating on new nodes.
      # Also backup memory after validation so it can be used for testing (since test edges are
      # strictly later in time than validation edges)
      tgn.memory.restore_memory(train_memory_backup)

    
    if USE_MEMORY:
      tgn.memory.__init_memory__()
    # Validate on unseen nodes
    # nn_val_ap, nn_val_auc = eval_edge_prediction(model=tgn,
    #                                                                     negative_edge_sampler=val_rand_sampler,
    #                                                                     data=new_node_val_data,
    #                                                                     n_neighbors=NUM_NEIGHBORS)
    nn_val_ap, nn_val_auc,nn_val_acc, nn_val_f1 = eval_edge_prediction_add_1(model=tgn,
                                                                        negative_edge_sampler=val_rand_sampler,
                                                                        data=new_node_val_data,
                                                                        n_neighbors=NUM_NEIGHBORS)

    if USE_MEMORY:
      # Restore memory we had at the end of validation
      tgn.memory.restore_memory(val_memory_backup)

    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    # train_losses.append(np.mean(m_loss))
    epoch_loss = np.mean(m_loss)
    train_losses.append(epoch_loss)
    swanlab.log({'train_loss_epoch': epoch_loss, 'epoch': epoch})

    
     # Log validation results to wandb
    swanlab.log({
      'val_auc': val_auc,
      'new_node_val_auc': nn_val_auc,
      'val_ap': val_ap,
      'new_node_val_ap': nn_val_ap,
      'val_acc': val_acc,
      'new_node_val_acc': nn_val_acc,
      'val_f1': val_f1,
      'new_node_val_f1': nn_val_f1,
      'epoch': epoch
    })
    # wandb.log({
    #   'val_auc': val_auc,
    #   'new_node_val_auc': nn_val_auc,
    #   'val_ap': val_ap,
    #   'new_node_val_ap': nn_val_ap,
    #   'val_acc': val_acc,
    #   'new_node_val_acc': nn_val_acc,
    #   'epoch': epoch
    # })
    # Save temporary results to disk
    pickle.dump({
      "val_aps": val_aps,
      "new_nodes_val_aps": new_nodes_val_aps,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    # logger.info(
    #   'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    # logger.info(
    #   'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
    # logger.info(
    #   'val acc: {}, new node val acc: {}'.format(val_acc, nn_val_acc))
    logger.info(
    'Validation statistics: Old nodes -- auc: {:.4f}, ap: {:.4f}, acc: {:.4f}, f1: {:.4f}'.format(
        val_auc, val_ap, val_acc, val_f1))
    logger.info(
        'Validation statistics: New nodes -- auc: {:.4f}, ap: {:.4f}, acc: {:.4f}, f1: {:.4f}'.format(
            nn_val_auc, nn_val_ap, nn_val_acc, nn_val_f1))


    # 保存当前 epoch 的嵌入
    #tgn.embedding_module.save_embeddings(EMBEDDING_DIR, epoch)
    
     # **检查是否为最佳模型**
    # W_AUC = 0.5
    # W_F1 = 0.5
    # mixed_loss = 1 - (W_AUC * val_auc + W_F1 * val_f1)

    # if mixed_loss < best_loss:
    #   best_loss = mixed_loss

    
    if val_auc > best_loss:
      best_loss = val_auc
      best_epoch = epoch
      early_stopper.best_epoch = epoch
      early_stopper.best_model_state = deepcopy(tgn.state_dict())
      torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
      logger.info(f'Best model saved at epoch {epoch}')
      #shutil.copyfile(f"{EMBEDDING_DIR}/embeddings_epoch_{epoch}.json", BEST_EMBEDDING_PATH)

    # **早停**
    if early_stopper.early_stop_check_raw(val_auc):
        logger.info(f'No improvement over {args.patience} epochs, stopping training at epoch {epoch}')
        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        tgn.load_state_dict(torch.load(MODEL_SAVE_PATH))
        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        tgn.eval()
        break
    else:
        torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

  

  # 训练结束后，加载最佳模型
  if best_epoch != -1:
    tgn.load_state_dict(early_stopper.best_model_state)
    logger.info(f'Loaded the best model from epoch {best_epoch} for testing')

  # 训练结束后，备份最佳模型的内存
  if USE_MEMORY:
    best_memory_backup = tgn.memory.backup_memory()
    
  if USE_MEMORY:
    tgn.memory.__init_memory__()

  ### Test
  tgn.embedding_module.neighbor_finder = full_ngh_finder
  # test_ap, test_auc = eval_edge_prediction(model=tgn,
  #                                                             negative_edge_sampler=test_rand_sampler,
  #                                                             data=test_data,
  #                                                             n_neighbors=NUM_NEIGHBORS)
  test_ap, test_auc,test_acc, test_f1 = eval_edge_prediction_add_1(model=tgn,
                                                              negative_edge_sampler=test_rand_sampler,
                                                              data=test_data,
                                                              n_neighbors=NUM_NEIGHBORS)

  if USE_MEMORY:
    tgn.memory.restore_memory(best_memory_backup)

  # Test on unseen nodes
  # nn_test_ap, nn_test_auc = eval_edge_prediction(model=tgn,
  #                                                                         negative_edge_sampler=nn_test_rand_sampler,
  #                                                                         data=new_node_test_data,
  #                                                                         n_neighbors=NUM_NEIGHBORS)
  nn_test_ap, nn_test_auc,nn_test_acc, nn_test_f1 = eval_edge_prediction_add_1(model=tgn,
                                                                          negative_edge_sampler=nn_test_rand_sampler,
                                                                          data=new_node_test_data,
                                                                          n_neighbors=NUM_NEIGHBORS)
  # Log test results to wandb
  # wandb.log({
  #   'test_auc': test_auc,
  #   'new_node_test_auc': nn_test_auc,
  #   'test_ap': test_ap,
  #   'new_node_test_ap': nn_test_ap,
  #   'test_acc': test_acc,
  #   'new_node_test_acc': nn_test_acc
  # })
  swanlab.log({
    'test_auc': test_auc,
    'new_node_test_auc': nn_test_auc,
    'test_ap': test_ap,
    'new_node_test_ap': nn_test_ap,
    'test_acc': test_acc,
    'new_node_test_acc': nn_test_acc,
    'test_f1': test_f1,
    'new_node_test_f1': nn_test_f1
  })
  # logger.info(
  #   'Test statistics: Old nodes -- auc: {}, ap: {},acc:{}'.format(test_auc, test_ap,test_acc))
  # logger.info(
  #   'Test statistics: New nodes -- auc: {}, ap: {},acc:{}'.format(nn_test_auc, nn_test_ap,nn_test_acc))
  
  logger.info(
    'Test statistics: Old nodes -- auc: {:.4f}, ap: {:.4f}, acc: {:.4f}, f1: {:.4f}'.format(
        test_auc, test_ap, test_acc, test_f1))
  logger.info(
      'Test statistics: New nodes -- auc: {:.4f}, ap: {:.4f}, acc: {:.4f}, f1: {:.4f}'.format(
          nn_test_auc, nn_test_ap, nn_test_acc, nn_test_f1))

  # Save results for this run
  pickle.dump({
    "val_aps": val_aps,
    "new_nodes_val_aps": new_nodes_val_aps,
    "test_ap": test_ap,
    "new_node_test_ap": nn_test_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  logger.info('Saving TGN model')
  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    tgn.memory.restore_memory(best_memory_backup)
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
  logger.info('TGN model saved')

  #tgn.embedding_module.save_embeddings(EMBEDDING_DIR, best_epoch)
  #shutil.copyfile(f"{EMBEDDING_DIR}/embeddings_epoch_{best_epoch}.json", BEST_EMBEDDING_PATH)
  #logger.info(f'All node embeddings saved to {BEST_EMBEDDING_PATH}')

# Finish the wandb run
# wandb.finish()
swanlab.finish()