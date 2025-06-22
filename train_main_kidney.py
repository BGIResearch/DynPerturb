from evaluation.evaluation import edge_prediction_eval1
from model.NetModel import NetModel
from utils.DataLoader import get_data_link1, compute_time_statistics
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
import math
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import os
from copy import deepcopy

# Set working directory and random seeds (modify as needed)
os.chdir("./")  # Use project root as working directory. Adjust if needed.
torch.manual_seed(0)
np.random.seed(0)

# Argument parser for command-line options
parser = argparse.ArgumentParser("NetModel self-supervised training")
parser.add_argument(
    "-d",
    "--data",
    type=str,
    help="Dataset name (e.g., wikipedia or reddit)",
    default="celltype32",
)
parser.add_argument("--bs", type=int, default=200, help="Batch size")
parser.add_argument(
    "--prefix", type=str, default="", help="Prefix for checkpoint naming"
)
parser.add_argument(
    "--n_degree", type=int, default=20, help="Number of neighbors to sample"
)
parser.add_argument("--n_head", type=int, default=2, help="Number of attention heads")
parser.add_argument("--n_epoch", type=int, default=100, help="Number of epochs")
parser.add_argument("--n_layer", type=int, default=1, help="Number of network layers")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument(
    "--patience", type=int, default=10, help="Patience for early stopping"
)
parser.add_argument("--n_runs", type=int, default=1, help="Number of runs")
parser.add_argument("--drop_out", type=float, default=0.1, help="Dropout probability")
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
parser.add_argument(
    "--node_dim", type=int, default=100, help="Node embedding dimension"
)
parser.add_argument(
    "--time_dim", type=int, default=100, help="Time embedding dimension"
)
parser.add_argument(
    "--backprop_every",
    type=int,
    default=1,
    help="Backpropagation frequency (in batches)",
)
parser.add_argument(
    "--use_memory",
    action="store_true",
    help="Enable node memory augmentation",
)
parser.add_argument(
    "--embedding_module",
    type=str,
    default="graph_attention",
    choices=["graph_attention", "graph_sum", "identity", "time"],
    help="Embedding module type",
)
parser.add_argument(
    "--message_function",
    type=str,
    default="identity",
    choices=["mlp", "identity"],
    help="Message function type",
)
parser.add_argument(
    "--memory_updater",
    type=str,
    default="gru",
    choices=["gru", "rnn"],
    help="Memory updater type",
)
parser.add_argument(
    "--aggregator", type=str, default="last", help="Message aggregator type"
)
parser.add_argument(
    "--memory_update_at_end",
    action="store_true",
    help="Update memory at the end of batch",
)
parser.add_argument(
    "--message_dim", type=int, default=100, help="Message dimension"
)
parser.add_argument(
    "--memory_dim",
    type=int,
    default=1000,
    help="Memory dimension per user",
)
parser.add_argument(
    "--different_new_nodes",
    action="store_true",
    help="Use disjoint new node sets for train and val",
)
parser.add_argument(
    "--uniform",
    action="store_true",
    help="Uniform sampling from temporal neighbors",
)
parser.add_argument(
    "--randomize_features",
    action="store_true",
    help="Randomize node features",
)
parser.add_argument(
    "--use_destination_embedding_in_message",
    action="store_true",
    default=True,
    help="Include destination embedding in message",
)
parser.add_argument(
    "--use_source_embedding_in_message",
    action="store_true",
    default=True,
    help="Include source embedding in message",
)
parser.add_argument(
    "--dyrep", action="store_true", help="Whether to run the dyrep model"
)

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# Set global variables from arguments
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
MODE = "link_prediction"

# Define output directories based on DATA name
MODEL_DIR = f"./saved_models/{DATA}/"
CHECKPOINT_DIR = f"./saved_checkpoints/{DATA}/"
EMBEDDING_DIR = f"./saved_embeddings/{DATA}/"
RESULTS_DIR = f"./results/{DATA}/"
LOG_DIR = f"./log/{DATA}/"

# Create output directories if they do not exist
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
Path(EMBEDDING_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = f"{MODEL_DIR}/{args.prefix}_best_model.pth"


def get_checkpoint_path(epoch):
    return f"{CHECKPOINT_DIR}/{args.prefix}_checkpoint_epoch_{epoch}.pth"


BEST_EMBEDDING_PATH = f"{EMBEDDING_DIR}/{args.prefix}_embeddings_best.json"

# Load data and features
(
    node_features,
    edge_features,
    full_data,
    train_data,
    val_data,
    test_data,
    new_node_val_data,
    new_node_test_data,
) = get_data_link1(
    DATA,
    different_new_nodes_between_val_and_test=args.different_new_nodes,
    randomize_features=args.randomize_features,
)

# Initialize neighbor finders for training and evaluation
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative edge samplers
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(
    new_node_val_data.sources, new_node_val_data.destinations, seed=1
)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(
    new_node_test_data.sources, new_node_test_data.destinations, seed=3
)

# Set device for computation
device_string = "cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)

# Compute time statistics for temporal encoding
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = (
    compute_time_statistics(
        full_data.sources, full_data.destinations, full_data.timestamps
    )
)

# Initialize best_memory_backup to avoid undefined variable
best_memory_backup = None

for i in range(args.n_runs):
    results_path = (
        f"{RESULTS_DIR}/{DATA}_{i}.pkl"
        if i > 0
        else f"{RESULTS_DIR}/{args.prefix}_{DATA}.pkl"
    )
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Initialize Model
    netmodel = NetModel(
        neighbor_finder=train_ngh_finder,
        node_features=node_features,
        edge_features=edge_features,
        device=device,
        n_layers=NUM_LAYER,
        n_heads=NUM_HEADS,
        dropout=DROP_OUT,
        use_memory=USE_MEMORY,
        message_dimension=MESSAGE_DIM,
        memory_dimension=MEMORY_DIM,
        memory_update_at_start=not args.memory_update_at_end,
        embedding_module_type=args.embedding_module,
        message_function=args.message_function,
        aggregator_type=args.aggregator,
        memory_updater_type=args.memory_updater,
        n_neighbors=NUM_NEIGHBORS,
        mean_time_shift_src=mean_time_shift_src,
        std_time_shift_src=std_time_shift_src,
        mean_time_shift_dst=mean_time_shift_dst,
        std_time_shift_dst=std_time_shift_dst,
        use_destination_embedding_in_message=args.use_destination_embedding_in_message,
        use_source_embedding_in_message=args.use_source_embedding_in_message,
        dyrep=args.dyrep,
    )
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(netmodel.parameters(), lr=LEARNING_RATE)
    netmodel = netmodel.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)
    idx_list = np.arange(num_instance)
    new_nodes_val_aps = []
    val_aps = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []
    early_stopper = EarlyStopMonitor(max_round=args.patience)
    best_loss = 0
    best_epoch = -1
    for epoch in range(NUM_EPOCH):
        start_epoch = time.time()
        # Training phase
        if USE_MEMORY:
            netmodel.memory.__init_memory__()
        netmodel.set_neighbor_finder(train_ngh_finder)
        m_loss = []
        for k in range(0, num_batch, args.backprop_every):
            loss = 0
            optimizer.zero_grad()
            for j in range(args.backprop_every):
                batch_idx = k + j
                if batch_idx >= num_batch:
                    continue
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)
                sources_batch, destinations_batch = (
                    train_data.sources[start_idx:end_idx],
                    train_data.destinations[start_idx:end_idx],
                )
                edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]
                size = len(sources_batch)
                _, negatives_batch = train_rand_sampler.sample(size)
                with torch.no_grad():
                    pos_label = torch.ones(size, dtype=torch.float, device=device)
                    neg_label = torch.zeros(size, dtype=torch.float, device=device)
                netmodel = netmodel.train()
                pos_prob, neg_prob = netmodel.compute_edge_probabilities(
                    sources_batch,
                    destinations_batch,
                    negatives_batch,
                    timestamps_batch,
                    edge_idxs_batch,
                    NUM_NEIGHBORS,
                )
                loss += criterion(pos_prob.squeeze(), pos_label) + criterion(
                    neg_prob.squeeze(), neg_label
                )
            loss /= args.backprop_every
            loss.backward()
            optimizer.step()
            step_loss = loss.item()
            m_loss.append(step_loss)
            if USE_MEMORY:
                netmodel.memory.detach_memory()
        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)
        # Validation phase
        netmodel.set_neighbor_finder(full_ngh_finder)
        if USE_MEMORY:
            train_memory_backup = netmodel.memory.backup_memory()
        val_ap, val_auc, val_acc, val_f1 = eval_edge_prediction_add_1(
            model=netmodel,
            negative_edge_sampler=val_rand_sampler,
            data=val_data,
            n_neighbors=NUM_NEIGHBORS,
        )
        if USE_MEMORY:
            val_memory_backup = netmodel.memory.backup_memory()
            netmodel.memory.restore_memory(train_memory_backup)
        if USE_MEMORY:
            netmodel.memory.__init_memory__()
        nn_val_ap, nn_val_auc, nn_val_acc, nn_val_f1 = edge_prediction_eval1(
            model=netmodel,
            negative_edge_sampler=val_rand_sampler,
            data=new_node_val_data,
            n_neighbors=NUM_NEIGHBORS,
        )
        if USE_MEMORY:
            netmodel.memory.restore_memory(val_memory_backup)
        new_nodes_val_aps.append(nn_val_ap)
        val_aps.append(val_ap)
        epoch_loss = np.mean(m_loss)
        train_losses.append(epoch_loss)
        # Save training/validation results with exception handling
        try:
            with open(results_path, "wb") as f:
                pickle.dump(
                    {
                        "val_aps": val_aps,
                        "new_nodes_val_aps": new_nodes_val_aps,
                        "train_losses": train_losses,
                        "epoch_times": epoch_times,
                        "total_epoch_times": total_epoch_times,
                    },
                    f,
                )
        except Exception as e:
            print(f"[Warning] Failed to save results: {e}")
        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)
        if val_auc > best_loss:
            best_loss = val_auc
            best_epoch = epoch
            early_stopper.best_epoch = epoch
            early_stopper.best_model_state = deepcopy(netmodel.state_dict())
            torch.save(netmodel.state_dict(), MODEL_SAVE_PATH)
        if early_stopper.early_stop_check_raw(val_auc):
            netmodel.eval()
            break
        else:
            torch.save(netmodel.state_dict(), get_checkpoint_path(epoch))
    if best_epoch != -1:
        netmodel.load_state_dict(early_stopper.best_model_state)
    if USE_MEMORY:
        try:
            best_memory_backup = netmodel.memory.backup_memory()
        except Exception as e:
            print(f"[Warning] Failed to backup memory: {e}")
    if USE_MEMORY:
        try:
            netmodel.memory.__init_memory__()
        except Exception as e:
            print(f"[Warning] Failed to init memory: {e}")
    # Test phase
    netmodel.embedding_module.neighbor_finder = full_ngh_finder
    test_ap, test_auc, test_acc, test_f1 = eval_edge_prediction_add_1(
        model=netmodel,
        negative_edge_sampler=test_rand_sampler,
        data=test_data,
        n_neighbors=NUM_NEIGHBORS,
    )
    if USE_MEMORY and best_memory_backup is not None:
        try:
            netmodel.memory.restore_memory(best_memory_backup)
        except Exception as e:
            print(f"[Warning] Failed to restore memory: {e}")
    nn_test_ap, nn_test_auc, nn_test_acc, nn_test_f1 = eval_edge_prediction_add_1(
        model=netmodel,
        negative_edge_sampler=nn_test_rand_sampler,
        data=new_node_test_data,
        n_neighbors=NUM_NEIGHBORS,
    )
    try:
        with open(results_path, "wb") as f:
            pickle.dump(
                {
                    "val_aps": val_aps,
                    "new_nodes_val_aps": new_nodes_val_aps,
                    "test_ap": test_ap,
                    "new_node_test_ap": nn_test_ap,
                    "epoch_times": epoch_times,
                    "train_losses": train_losses,
                    "total_epoch_times": total_epoch_times,
                },
                f,
            )
    except Exception as e:
        print(f"[Warning] Failed to save test results: {e}")
    if USE_MEMORY and best_memory_backup is not None:
        try:
            netmodel.memory.restore_memory(best_memory_backup)
        except Exception as e:
            print(f"[Warning] Failed to restore memory: {e}")
    try:
        torch.save(netmodel.state_dict(), MODEL_SAVE_PATH)
    except Exception as e:
        print(f"[Warning] Failed to save model: {e}")
