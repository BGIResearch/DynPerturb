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
import logging
from evaluation.evaluation import edge_prediction_eval_link
from model.DynPerturbModel import DynPerturbModel
from utils.DataLoader import get_data, compute_time_statistics
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder

# Set working directory and random seeds for reproducibility
os.chdir("./")
torch.manual_seed(0)
np.random.seed(0)

# Parse command-line arguments for model and training configuration
parser = argparse.ArgumentParser("DynPerturbModel self-supervised training")
parser.add_argument("-d", "--data", type=str, help="Dataset name (e.g., wikipedia or reddit)", default="celltype32")
parser.add_argument("--bs", type=int, default=200, help="Batch size")
parser.add_argument("--prefix", type=str, default="", help="Prefix for checkpoint naming")
parser.add_argument("--n_degree", type=int, default=20, help="Number of neighbors to sample")
parser.add_argument("--n_head", type=int, default=2, help="Number of attention heads")
parser.add_argument("--n_epoch", type=int, default=100, help="Number of epochs")
parser.add_argument("--n_layer", type=int, default=1, help="Number of network layers")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
parser.add_argument("--n_runs", type=int, default=1, help="Number of runs")
parser.add_argument("--drop_out", type=float, default=0.1, help="Dropout probability")
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
parser.add_argument("--node_dim", type=int, default=100, help="Node embedding dimension")
parser.add_argument("--time_dim", type=int, default=100, help="Time embedding dimension")
parser.add_argument("--backprop_every", type=int, default=1, help="Backpropagation frequency (in batches)")
parser.add_argument("--use_memory", action="store_true", help="Enable node memory augmentation")
parser.add_argument("--embedding_module", type=str, default="graph_attention", choices=["graph_attention", "graph_sum", "identity", "time"], help="Embedding module type")
parser.add_argument("--message_function", type=str, default="identity", choices=["mlp", "identity"], help="Message function type")
parser.add_argument("--memory_updater", type=str, default="gru", choices=["gru", "rnn"], help="Memory updater type")
parser.add_argument("--aggregator", type=str, default="last", help="Message aggregator type")
parser.add_argument("--memory_update_at_end", action="store_true", help="Update memory at the end of batch")
parser.add_argument("--message_dim", type=int, default=100, help="Message dimension")
parser.add_argument("--memory_dim", type=int, default=1000, help="Memory dimension per user")
parser.add_argument("--different_new_nodes", action="store_true", help="Use disjoint new node sets for train and val")
parser.add_argument("--uniform", action="store_true", help="Uniform sampling from temporal neighbors")
parser.add_argument("--randomize_features", action="store_true", help="Randomize node features")
parser.add_argument("--use_destination_embedding_in_message", action="store_true", default=True, help="Include destination embedding in message")
parser.add_argument("--use_source_embedding_in_message", action="store_true", default=True, help="Include source embedding in message")
parser.add_argument("--dyrep", action="store_true", help="Whether to run the dyrep model")

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
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
MODE = "link_prediction"

# Define output directories for models, checkpoints, embeddings, results, and logs
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

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'./train.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

def get_checkpoint_path(epoch):
    return f"{CHECKPOINT_DIR}/{args.prefix}_checkpoint_epoch_{epoch}.pth"

BEST_EMBEDDING_PATH = f"{EMBEDDING_DIR}/{args.prefix}_embeddings_best.json"

# Load data and features
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, num_nodes = get_data(DATA, different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)

# Initialize neighbor finders for training and evaluation
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative edge samplers for link prediction
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

# Set device for computation
device_string = "cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)

# Compute time statistics for temporal encoding
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

# Initialize best_memory_backup to avoid undefined variable
best_memory_backup = None

# Ensure USE_MEMORY is defined in the global scope
USE_MEMORY = args.use_memory

# Main training and validation loop
for i in range(args.n_runs):
    results_path = f"{RESULTS_DIR}/{DATA}_{i}.pkl" if i > 0 else f"{RESULTS_DIR}/{args.prefix}_{DATA}.pkl"
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Initialize model for link prediction
    dynperturb_model = DynPerturbModel(
        num_nodes = num_nodes, neighbor_finder = train_ngh_finder, node_features = node_features,
        edge_features = edge_features, device = device, n_layers = NUM_LAYER, n_heads = NUM_HEADS, dropout = DROP_OUT,
        use_memory = USE_MEMORY, message_dimension = MESSAGE_DIM, memory_dimension = MEMORY_DIM, memory_update_at_start = not args.memory_update_at_end,
        embedding_module_type = args.embedding_module, message_function = args.message_function, aggregator_type = args.aggregator,
        memory_updater_type = args.memory_updater, n_neighbors = NUM_NEIGHBORS, mean_time_shift_src = mean_time_shift_src,
        std_time_shift_src = std_time_shift_src, mean_time_shift_dst = mean_time_shift_dst, std_time_shift_dst = std_time_shift_dst,
        use_destination_embedding_in_message = args.use_destination_embedding_in_message,
        use_source_embedding_in_message = args.use_source_embedding_in_message, dyrep = args.dyrep)
    
    # Define loss function and optimizer for training
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(dynperturb_model.parameters(), lr = LEARNING_RATE)
    dynperturb_model = dynperturb_model.to(device)

    # Prepare training statistics and early stopping
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
    best_loss = 0
    best_epoch = -1

    # Epoch training loop
    for epoch in range(NUM_EPOCH):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCH}")
        start_epoch = time.time()
        logger.info('start {} epoch'.format(epoch))
        
        # Training phase
        if USE_MEMORY:
            dynperturb_model.memory.__init_memory__()  # Re-initialize memory at the start of each epoch
        
        dynperturb_model.set_neighbor_finder(train_ngh_finder)  # Set neighbor finder for training
        m_loss = []  # Store batch losses for this epoch
        
        # Iterate over batches with backpropagation frequency
        for k in range(0, num_batch, args.backprop_every):
            loss = 0
            optimizer.zero_grad()
            
            # Accumulate gradients over backprop_every batches
            for j in range(args.backprop_every):
                batch_idx = k + j
                if batch_idx >= num_batch:
                    continue
                
                # Prepare Batch Data
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)
                sources_batch = train_data.sources[start_idx:end_idx]
                destinations_batch = train_data.destinations[start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]
                
                size = len(sources_batch)
                _, negatives_batch = train_rand_sampler.sample(size)  # Sample negative edges
                
                with torch.no_grad():
                    pos_label = torch.ones(size, dtype=torch.float, device=device)
                    neg_label = torch.zeros(size, dtype=torch.float, device=device)
                
                # Forward Pass 
                dynperturb_model = dynperturb_model.train()
                pos_prob, neg_prob = dynperturb_model.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
                loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
                
            loss /= args.backprop_every
            loss.backward()
            optimizer.step()
            step_loss = loss.item()
            m_loss.append(step_loss)
            if USE_MEMORY:
                dynperturb_model.memory.detach_memory()
                
        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        # Print average training loss for this epoch
        avg_train_loss = np.mean(m_loss)
        print(f"Training Loss: {avg_train_loss:.4f} (Epoch Time: {epoch_time:.2f}s)")

        # Validation phase
        dynperturb_model.set_neighbor_finder(full_ngh_finder)
        if USE_MEMORY:
            train_memory_backup = dynperturb_model.memory.backup_memory()
            
        val_ap, val_auc, val_acc, val_f1 = edge_prediction_eval_link(model=dynperturb_model, negative_edge_sampler=val_rand_sampler, data=val_data, n_neighbors=NUM_NEIGHBORS)
        if USE_MEMORY:
            val_memory_backup = dynperturb_model.memory.backup_memory()
            dynperturb_model.memory.restore_memory(train_memory_backup)
            
        if USE_MEMORY:
            dynperturb_model.memory.__init_memory__()
        nn_val_ap, nn_val_auc, nn_val_acc, nn_val_f1 = edge_prediction_eval_link(model=dynperturb_model, negative_edge_sampler=val_rand_sampler, data=new_node_val_data, n_neighbors=NUM_NEIGHBORS)
        
        if USE_MEMORY:
            dynperturb_model.memory.restore_memory(val_memory_backup)
            
        new_nodes_val_aps.append(nn_val_ap)
        val_aps.append(val_ap)
        epoch_loss = np.mean(m_loss)
        train_losses.append(epoch_loss)

        # Print validation performance
        print(f"Validation AP: {val_ap:.4f} | AUC: {val_auc:.4f} | Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")
        try:
            with open(results_path, "wb") as f:
                pickle.dump({"val_aps": val_aps, "new_nodes_val_aps": new_nodes_val_aps, "train_losses": train_losses, "epoch_times": epoch_times, "total_epoch_times": total_epoch_times}, f)
        except Exception as e:
            print(f"[Warning] Failed to save results: {e}")
            
        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)
        
        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('Validation statistics: Old nodes -- auc: {:.4f}, ap: {:.4f}, acc: {:.4f}, f1: {:.4f}'.format(val_auc, val_ap, val_acc, val_f1))
        logger.info('Validation statistics: New nodes -- auc: {:.4f}, ap: {:.4f}, acc: {:.4f}, f1: {:.4f}'.format(nn_val_auc, nn_val_ap, nn_val_acc, nn_val_f1))
        
        # Save Best Model and Early Stopping
        if val_auc > best_loss:
            best_loss = val_auc
            best_epoch = epoch
            early_stopper.best_epoch = epoch
            early_stopper.best_model_state = deepcopy(dynperturb_model.state_dict())
            torch.save(dynperturb_model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f'Best model saved at epoch {epoch}')
            
        # Check early stopping condition
        if early_stopper.early_stop_check_raw(val_auc):
            logger.info(f'No improvement over {args.patience} epochs, stopping training at epoch {epoch}')
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            dynperturb_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            dynperturb_model.eval()
            break
        else:
            torch.save(dynperturb_model.state_dict(), get_checkpoint_path(epoch))

    if best_epoch != -1:
        dynperturb_model.load_state_dict(early_stopper.best_model_state)
        logger.info(f'Loaded the best model from epoch {best_epoch} for testing')
        
    if USE_MEMORY:
        try:
            best_memory_backup = dynperturb_model.memory.backup_memory()
        except Exception as e:
            print(f"[Warning] Failed to backup memory: {e}")
            
    if USE_MEMORY:
        try:
            dynperturb_model.memory.__init_memory__()
        except Exception as e:
            print(f"[Warning] Failed to init memory: {e}")
            
    # Test phase
    dynperturb_model.embedding_module.neighbor_finder = full_ngh_finder
    test_ap, test_auc, test_acc, test_f1 = edge_prediction_eval_link(model=dynperturb_model, negative_edge_sampler=test_rand_sampler, data=test_data, n_neighbors=NUM_NEIGHBORS)
    logger.info('Test statistics: Old nodes -- auc: {:.4f}, ap: {:.4f}, acc: {:.4f}, f1: {:.4f}'.format(test_auc, test_ap, test_acc, test_f1))
    logger.info('Test statistics: New nodes -- auc: {:.4f}, ap: {:.4f}, acc: {:.4f}, f1: {:.4f}'.format(nn_test_auc, nn_test_ap, nn_test_acc, nn_test_f1))
    
    # Restore Memory for Test on New Nodes (if needed) 
    if USE_MEMORY and best_memory_backup is not None:
        try:
            dynperturb_model.memory.restore_memory(best_memory_backup)
        except Exception as e:
            print(f"[Warning] Failed to restore memory: {e}")
    
    # Evaluate on New Node Test Set
    nn_test_ap, nn_test_auc, nn_test_acc, nn_test_f1 = edge_prediction_eval_link(model=dynperturb_model, negative_edge_sampler=nn_test_rand_sampler, data=new_node_test_data, n_neighbors=NUM_NEIGHBORS)
    try:
        with open(results_path, "wb") as f:
            pickle.dump({"val_aps": val_aps, "new_nodes_val_aps": new_nodes_val_aps, "test_ap": test_ap, "new_node_test_ap": nn_test_ap, "epoch_times": epoch_times, "train_losses": train_losses, "total_epoch_times": total_epoch_times}, f)
            logger.info('Saving TGN model')
    except Exception as e:
        print(f"[Warning] Failed to save test results: {e}")
        
    # Restore Memory Again (if needed)
    if USE_MEMORY and best_memory_backup is not None:
        try:
            dynperturb_model.memory.restore_memory(best_memory_backup)
        except Exception as e:
            print(f"[Warning] Failed to restore memory: {e}")
            
    # Save Final Model
    try:
        torch.save(dynperturb_model.state_dict(), MODEL_SAVE_PATH)
        logger.info('TGN model saved')
    except Exception as e:
        print(f"[Warning] Failed to save model: {e}")
