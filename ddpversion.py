from evaluation.evaluation import (
    node_classification_eval,
    edge_prediction_eval2,
)

from model.NetModel import NetModel

from utils.DataLoader import get_data_ddp1, compute_time_statistics

from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder


import math
import time
import sys
import random
import argparse
from pathlib import Path
import torch
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Set working directory (use relative path or comment out if not needed)

os.chdir("./")  # Use project root as working directory. Adjust if needed.

# Argument parser for command-line options
parser = argparse.ArgumentParser("NetModel self-supervised training")
parser.add_argument(
    "-d",
    "--data",
    type=str,
    help="Dataset name (e.g., wikipedia or reddit)",
    default="HumanBone",
)
parser.add_argument("--bs", type=int, default=64, help="Batch size")
parser.add_argument(
    "--prefix", type=str, default="", help="Prefix for checkpoint naming"
)
parser.add_argument(
    "--n_degree", type=int, default=20, help="Number of neighbors to sample"
)
parser.add_argument("--n_head", type=int, default=2, help="Number of attention heads")
parser.add_argument("--n_epoch", type=int, default=100, help="Number of epochs")
parser.add_argument("--n_layer", type=int, default=1, help="Number of network layers")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
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
    choices=["gru", "rnn", "lstm"],
    help="Memory updater type",
)
parser.add_argument(
    "--aggregator", type=str, default="last", help="Message aggregator type"
)
parser.add_argument(
    "--memory_update_at_end",
    action="store_true",
    default=True,
    help="Update memory at the end of batch",
)
parser.add_argument("--message_dim", type=int, default=100, help="Message dimension")
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
parser.add_argument("--n_neg", type=int, default=1)
parser.add_argument(
    "--use_validation",
    action="store_true",
    default=True,
    help="Use validation set",
)
parser.add_argument("--new_node", action="store_true", help="Model new node")

try:
    args = parser.parse_args()
except Exception:
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
MODE = "node_classification"

# Create directories for saving models and checkpoints
Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = (
    f"./saved_models/{args.prefix}-{args.data}" + "node-classification.pth"
)


def get_checkpoint_path(epoch):
    return (
        f"./saved_checkpoints/{args.prefix}-{args.data}-{epoch}node-classification.pth"
    )


# Output directory for figures
output_dir = "./figs"
os.makedirs(output_dir, exist_ok=True)

num_classes = 8

# Load data and features
(
    full_data,
    node_features,
    edge_features,
    train_data,
    val_data,
    test_data,
    new_node_val_data,
    new_node_test_data,
) = get_data_ddp1(DATA, use_validation=args.use_validation)

max_idx = max(full_data.unique_nodes)

# Initialize neighbor finders
train_ngh_finder = get_neighbor_finder(
    train_data, uniform=UNIFORM, max_node_idx=max_idx
)

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

# Set device for training
device_string = "cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)

# Compute time statistics for temporal encoding
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = (
    compute_time_statistics(
        full_data.sources, full_data.destinations, full_data.timestamps
    )
)

for i in range(args.n_runs):
    results_path = (
        "./results/{}_node_classification_{}.pkl".format(args.prefix, i)
        if i > 0
        else "./results/{}_node_classification.pkl".format(args.prefix)
    )
    Path("./results/").mkdir(parents=True, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # Set current process GPU
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Initialize distributed process group
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

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
        num_classes=num_classes,
        mode=MODE,
    )

    netmodel = netmodel.to(device)
    netmodel = DDP(
        netmodel,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    optimizer = torch.optim.Adam(netmodel.parameters(), lr=args.lr, weight_decay=1e-5)

    val_aucs = []
    train_losses = []

    best_combined_auc = -float("inf")
    best_model_epoch = None

    from sklearn.utils.class_weight import compute_class_weight

    weights = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=train_data.labels
    )
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    early_stopper = EarlyStopMonitor(max_round=args.patience)
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_data.sources),
        torch.from_numpy(train_data.destinations),
        torch.from_numpy(train_data.timestamps),
        torch.from_numpy(train_data.edge_idxs),
        torch.from_numpy(train_data.labels),
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    # Initialize val_memory_backup to avoid undefined variable risk
    val_memory_backup = None

    for epoch in range(args.n_epoch):
        start_epoch = time.time()

        # Initialize memory at each epoch if enabled
        if USE_MEMORY:
            netmodel.module.memory.__init_memory__()

        # Set neighbor finder for training
        netmodel.module.set_neighbor_finder(train_ngh_finder)

        optimizer.zero_grad()
        netmodel = netmodel.train()
        m_loss = []

        for batch in train_loader:
            sources_batch, dest_batch, ts_batch, edge_idxs_batch, labels_batch = batch
            sources_batch = sources_batch.to(device, non_blocking=True)
            destinations_batch = dest_batch.to(device, non_blocking=True)
            timestamps_batch = ts_batch.to(device, non_blocking=True)
            edge_idxs_batch = edge_idxs_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            size = len(sources_batch)

            try:
                _, negatives_batch = train_rand_sampler.sample(size)
            except Exception as e:
                print(f"Error in negative sampling: {e}")
                continue

            optimizer.zero_grad()

            try:
                pos_score, neg_score, node_classification_logits = netmodel(
                    sources_batch,
                    destinations_batch,
                    negatives_batch,
                    timestamps_batch,
                    edge_idxs_batch,
                )
            except Exception as e:
                print(f"Error in model forward: {e}")
                continue

            # Link prediction loss (binary cross-entropy)
            pos_labels = torch.ones(
                len(sources_batch), dtype=torch.float, device=device
            )
            neg_labels = torch.zeros(
                len(sources_batch), dtype=torch.float, device=device
            )
            pos_score = pos_score.squeeze(dim=-1)
            neg_score = neg_score.squeeze(dim=-1)
            link_pred_loss = torch.nn.BCELoss()(
                pos_score, pos_labels
            ) + torch.nn.BCELoss()(neg_score, neg_labels)

            # Node classification loss (cross-entropy)
            labels_batch = torch.tensor(labels_batch, dtype=torch.long).to(device)
            if node_classification_logits is not None:
                node_classification_loss = torch.nn.CrossEntropyLoss(weight=weights)(
                    node_classification_logits, labels_batch
                )
            else:
                node_classification_loss = 0

            # Total loss is the sum of both tasks
            total_loss = link_pred_loss + node_classification_loss
            try:
                total_loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"Error in backward/optimizer step: {e}")
                continue

            # Aggregate loss across DDP processes
            loss_tensor = torch.tensor(total_loss.item(), device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor / dist.get_world_size()

            m_loss.append(avg_loss.item())

        train_losses.append(np.mean(m_loss))

        if USE_MEMORY:
            try:
                netmodel.module.memory.detach_memory()
            except Exception as e:
                print(f"Error in detach_memory: {e}")

        # Validation phase
        netmodel.module.set_neighbor_finder(full_ngh_finder)

        if USE_MEMORY:
            try:
                train_memory_backup = netmodel.module.memory.backup_memory()
            except Exception as e:
                print(f"Error in backup_memory: {e}")
                train_memory_backup = None

        # Evaluate link prediction on validation set
        (
            val_ap_link_prediction,
            val_auc_link_prediction,
            val_precision_link_prediction,
            val_recall_link_prediction,
            val_f1_link_prediction,
            val_acc_link_prediction,
        ) = edge_prediction_eval2(
            netmodel,
            val_rand_sampler,
            val_data,
            NUM_NEIGHBORS,eval
            BATCH_SIZE,
        )

        # Evaluate node classification on validation set
        (
            val_auc_node_classification,
            precision,
            recall,
            f1,
            support,
            pred_prob,
            true_labels_val,
        ) = node_classification_eval(
            netmodel,
            val_data,
            full_data.edge_idxs,
            BATCH_SIZE,
            NUM_NEIGHBORS,
            num_classes,
        )

        if USE_MEMORY:
            try:
                val_memory_backup = netmodel.module.memory.backup_memory()
                if train_memory_backup is not None:
                    netmodel.module.memory.restore_memory(train_memory_backup)
            except Exception as e:
                print(f"Error in val/train memory backup/restore: {e}")

        (
            nn_val_ap,
            nn_val_auc,
            nn_val_precision,
            nn_val_recall,
            nn_val_f1,
            nn_val_acc,
        ) = edge_prediction_eval2(
            model=netmodel,
            negative_edge_sampler=val_rand_sampler,
            data=new_node_val_data,
            n_neighbors=NUM_NEIGHBORS,
        )

        if USE_MEMORY:
            try:
                netmodel.module.memory.restore_memory(val_memory_backup)
            except Exception as e:
                print(f"Error in restore_memory: {e}")

        val_f1_node_classification_mean = np.mean(f1)
        combined_auc = (val_auc_link_prediction + val_f1_node_classification_mean) / 2

        # Early stopping check (only on main process)
        if dist.get_rank() == 0:
            stop_flag = early_stopper.early_stop_check(combined_auc)
        else:
            stop_flag = False

        stop_tensor = torch.tensor(int(stop_flag), device=device)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item() == 1:
            break

        if dist.get_rank() == 0 and combined_auc > best_combined_auc:
            best_combined_auc = combined_auc
            best_model_epoch = epoch
            try:
                torch.save(netmodel.module.state_dict(), get_checkpoint_path(epoch))
            except Exception as e:
                print(f"Error saving model: {e}")

    if dist.get_rank() == 0:
        try:
            torch.save(netmodel.module.state_dict(), get_checkpoint_path(epoch))
        except Exception as e:
            print(f"Error saving model: {e}")

    dist.barrier()

    # Synchronize best_model_epoch across all processes
    best_model_epoch_tensor = torch.tensor(
        best_model_epoch if dist.get_rank() == 0 else -1, device=device
    )
    dist.broadcast(best_model_epoch_tensor, src=0)
    best_model_epoch = best_model_epoch_tensor.item()
    if best_model_epoch == -1:
        raise RuntimeError(
            "No best model was saved. Check if training diverged or AUC is invalid."
        )

    try:
        netmodel.module.load_state_dict(
            torch.load(get_checkpoint_path(best_model_epoch))
        )
    except Exception as e:
        print(f"Error loading best model: {e}")

    if USE_MEMORY:
        try:
            val_memory_backup = netmodel.module.memory.backup_memory()
        except Exception as e:
            print(f"Error in backup_memory: {e}")
            val_memory_backup = None

    netmodel.module.embedding_module.neighbor_finder = full_ngh_finder

    (
        test_ap_link_prediction,
        test_auc_link_prediction,
        test_precision_link_prediction,
        test_recall_link_prediction,
        test_f1_link_prediction,
        test_acc_link_prediction,
    ) = eval_edge_prediction_add_ddp_new(
        netmodel, test_rand_sampler, test_data, NUM_NEIGHBORS, BATCH_SIZE
    )

    (
        test_auc_node_classification,
        precision,
        recall,
        f1,
        support,
        pred_prob,
        true_labels,
    ) = eval_node_classification_ddp_new(
        netmodel, test_data, full_data.edge_idxs, BATCH_SIZE, NUM_NEIGHBORS, num_classes
    )

    # Restore memory if enabled
    if USE_MEMORY:
        try:
            netmodel.module.memory.restore_memory(val_memory_backup)
        except Exception as e:
            print(f"[Warning] Failed to restore memory: {e}")

    # Evaluate edge prediction for new node test set
    (
        nn_test_ap,
        nn_test_auc,
        nn_test_precision,
        nn_test_recall,
        nn_test_f1,
        nn_test_acc,
    ) = eval_edge_prediction_add_ddp_new(
        model=netmodel,
        negative_edge_sampler=nn_test_rand_sampler,
        data=new_node_test_data,
        n_neighbors=NUM_NEIGHBORS,
    )

    # Plot and save confusion matrix and AUC bar chart
    try:
        y_true = true_labels if "true_labels" in locals() else []
        pred_prob_safe = (
            pred_prob
            if "pred_prob" in locals()
            else np.zeros((len(y_true), num_classes))
        )
        y_pred = np.argmax(pred_prob_safe, axis=1)
        class_names = [f"Class {i}" for i in range(num_classes)]

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{args.prefix}-{args.data}_confusion_matrix.pdf")
        )
        try:
            plt.close()
        except Exception:
            pass

        cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix (Normalized)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir, f"{args.prefix}-{args.data}_confusion_matrix_normalized.pdf"
            )
        )
        try:
            plt.close()
        except Exception:
            pass

        plt.figure(figsize=(10, 5))
        auc_bar = (
            test_auc_node_classification
            if "test_auc_node_classification" in locals()
            else np.zeros(num_classes)
        )
        plt.bar(class_names, auc_bar)
        plt.ylabel("AUC Score")
        plt.title("Per-Class AUC")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{args.prefix}-{args.data}_auc_bar_chart.pdf")
        )
        try:
            plt.close()
        except Exception:
            pass
    except Exception as e:
        print(f"[Warning] Plotting failed: {e}")

    # Restore memory again if enabled
    if USE_MEMORY:
        try:
            netmodel.module.memory.restore_memory(val_memory_backup)
        except Exception as e:
            print(f"[Warning] Failed to restore memory: {e}")
    torch.save(netmodel.module.state_dict(), MODEL_SAVE_PATH)
