from model.NetModel import NetModel
from utils.DataLoader import get_data_link2, compute_time_statistics
from utils.utils import RandEdgeSampler, get_neighbor_finder
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import os
import json
from collections import defaultdict

# Set environment variables for debugging and disabling pixi collection
os.environ["SWANLAB_REQUIREMENTS"] = "off"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set working directory (modify as needed)
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
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
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
    help="Update memory at the end of batch",
)
parser.add_argument("--message_dim", type=int, default=100, help="Message dimension")
parser.add_argument(
    "--memory_dim",
    type=int,
    default=1000,
    help="Memory dimension per user",
)
parser.add_argument("--num_classes", type=int, default=4, help="Number of classes")
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
    help="Include destination embedding in message",
)
parser.add_argument(
    "--use_source_embedding_in_message",
    action="store_true",
    help="Include source embedding in message",
)
parser.add_argument(
    "--use_validation",
    action="store_true",
    default=True,
    help="Use validation set",
)

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
NumClasses = args.num_classes
MODE = "link-prediction"

# Define output directories based on DATA name
MODEL_DIR = f"./saved_models/{DATA}/"
CHECKPOINT_DIR = f"./saved_checkpoints/{DATA}/"
EMBEDDING_DIR = f"./saved_embeddings/{DATA}/"
RESULTS_DIR = f"./results/{DATA}/"

# Create output directories if they do not exist
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
Path(EMBEDDING_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


MODEL_SAVE_PATH = f"{MODEL_DIR}/best_model.pth"
get_checkpoint_path = lambda epoch: f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch}.pth"
RESULTS_PATH = f"{RESULTS_DIR}/results.pkl"
BEST_EMBEDDING_PATH = f"{EMBEDDING_DIR}/embeddings_best.json"

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
) = get_data_link2(DATA, use_validation=args.use_validation)

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

# Set model save path (modify as needed)
MODEL_SAVE_PATH = "/home/share/huadjyin/home/s_qinhua2/02code/netmodel-master/kidney/saved_models/PT-S2-D/PT-S2-D_only_link_best_model.pth"
if not os.path.exists(MODEL_SAVE_PATH):
    raise FileNotFoundError(
        f"Best model not found at {MODEL_SAVE_PATH}, please train first."
    )

# Initialize NetModel
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
    num_classes=NumClasses,
    mode=MODE,
)

netmodel = netmodel.to(device)
netmodel.load_state_dict(torch.load(MODEL_SAVE_PATH), strict=False)
netmodel.eval()

# Initialize memory if enabled
if USE_MEMORY:
    netmodel.memory.__init_memory__()
netmodel.set_neighbor_finder(full_ngh_finder)

saved_embeddings = defaultdict(list)

# Prepare all events in the graph (sorted by time)
sources = full_data.sources
destinations = full_data.destinations
timestamps = full_data.timestamps
edge_idxs = full_data.edge_idxs
num_events = len(sources)

# Compute and save embeddings in batches with exception handling
with torch.no_grad():
    try:
        for start in range(0, num_events, BATCH_SIZE):
            end = min(start + BATCH_SIZE, num_events)
            src = torch.from_numpy(sources[start:end]).to(device)
            dst = torch.from_numpy(destinations[start:end]).to(device)
            ts = torch.from_numpy(timestamps[start:end]).to(device)
            eid = torch.from_numpy(edge_idxs[start:end]).to(device)
            neg = dst.clone()
            emb_src, emb_dst, _ = netmodel.compute_temporal_embeddings(
                src, dst, neg, ts, eid
            )
            for i in range(len(src)):
                s_id = int(src[i])
                d_id = int(dst[i])
                t_val = float(ts[i])
                saved_embeddings[s_id].append(
                    {
                        "timestamp": t_val,
                        "embedding": emb_src[i].detach().cpu().numpy().tolist(),
                    }
                )
                saved_embeddings[d_id].append(
                    {
                        "timestamp": t_val,
                        "embedding": emb_dst[i].detach().cpu().numpy().tolist(),
                    }
                )
    except Exception as e:
        print(f"[Warning] Embedding computation failed: {e}")

# Save embeddings as JSON with numpy type handling and exception protection
Path(EMBEDDING_DIR).mkdir(parents=True, exist_ok=True)
save_path = os.path.join(EMBEDDING_DIR, f"embeddings_{args.prefix}.json")
try:
    with open(save_path, "w") as f:
        json.dump(saved_embeddings, f, default=str)
except Exception as e:
    print(f"[Warning] Failed to save embeddings: {e}")
