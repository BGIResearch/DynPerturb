import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
)
import torch.distributed as dist
import math


def edge_prediction_eval_link(
    model, negative_edge_sampler, data, n_neighbors, batch_size=200
):
    """
    Evaluate edge prediction (binary classification) for a batch of edges (inductive setting).
    Returns average precision, AUC, accuracy, and F1 score.
    """
    assert negative_edge_sampler.seed is not None  # Ensure sampler is seeded
    negative_edge_sampler.reset_random_state()  # Reset sampler state

    val_ap, val_auc, val_acc, val_f1 = [], [], [], []  # Store metrics for each batch

    with torch.no_grad():
        model = model.eval()  # Set model to evaluation mode
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            # Prepare batch data for evaluation
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]

            size = len(sources_batch)
            # Sample negative edges for evaluation
            _, negative_samples = negative_edge_sampler.sample(size)

            # Compute probabilities for positive and negative edges
            pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch,destinations_batch,negative_samples,timestamps_batch,edge_idxs_batch,n_neighbors,)

            # Concatenate results and compute metrics
            y_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            y_true = np.concatenate([np.ones(size), np.zeros(size)])
            y_pred = (y_score >= 0.5).astype(int)

            val_ap.append(average_precision_score(y_true, y_score))
            val_auc.append(roc_auc_score(y_true, y_score))
            val_acc.append(accuracy_score(y_true, y_pred))
            val_f1.append(f1_score(y_true, y_pred))

    return (np.mean(val_ap), np.mean(val_auc), np.mean(val_acc), np.mean(val_f1))


def node_classification_eval(
    netmodel, data, edge_idxs, batch_size, n_neighbors, num_classes
):
    """
    Evaluate node classification in distributed setting.
    Returns per-class AUC, precision, recall, F1, support, predicted
    probabilities, and true labels.
    """
    # local_pred_prob and local_true_labels store predictions and labels for this process
    local_pred_prob = []
    local_true_labels = []

    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        netmodel.eval()  # Set model to evaluation mode

        for k in range(num_batch):
            # Prepare batch data for evaluation
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx:e_idx]
            labels_batch = data.labels[s_idx:e_idx]

            # Compute temporal embeddings for source and destination nodes
            source_embedding, destination_embedding, _ = (
                netmodel.module.compute_temporal_embeddings(
                    sources_batch,
                    destinations_batch,
                    destinations_batch,
                    timestamps_batch,
                    edge_idxs_batch,
                    n_neighbors,
                )
            )

            # Get logits and probabilities for node classification
            logits = netmodel.module.node_classification_decoder(source_embedding)
            prob = torch.softmax(logits, dim=-1)

            local_pred_prob.append(prob.cpu())
            local_true_labels.append(torch.tensor(labels_batch, dtype=torch.long))

    # Gather predictions and labels from all processes
    local_pred_prob = torch.cat(local_pred_prob, dim=0).cpu().numpy()
    local_true_labels = torch.cat(local_true_labels, dim=0).cpu().numpy()

    gathered_probs = [None] * dist.get_world_size()
    gathered_labels = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_probs, local_pred_prob.tolist())
    dist.all_gather_object(gathered_labels, local_true_labels.tolist())

    # Concatenate all predictions and labels from all processes
    all_pred_prob = np.concatenate([np.array(p) for p in gathered_probs], axis=0)
    all_true_labels = np.concatenate(
        [np.array(label) for label in gathered_labels], axis=0
    )

    try:
        # Compute per-class and weighted AUC
        auc_roc = roc_auc_score(
            all_true_labels, all_pred_prob, average=None, multi_class="ovr"
        )
    except ValueError as e:
        print(f"[Warning] AUC error: {e}")
        auc_roc = np.full(num_classes, np.nan)

    # Compute precision, recall, F1, support for each class and weighted
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true_labels, np.argmax(all_pred_prob, axis=1), average=None, zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, weighted_support = (
        precision_recall_fscore_support(
            all_true_labels,
            np.argmax(all_pred_prob, axis=1),
            average="weighted",
            zero_division=0,
        )
    )

    return (auc_roc, precision, recall, f1, support, all_pred_prob, all_true_labels)


def edge_prediction_eval_ddp(
    model, negative_edge_sampler, data, n_neighbors, batch_size=200
):
    """
    Evaluate edge prediction in distributed setting (transductive setting).
    Returns AP, AUC, precision, recall, F1, accuracy.
    """
    assert negative_edge_sampler.seed is not None  # Ensure sampler is seeded
    negative_edge_sampler.reset_random_state()  # Reset sampler state

    device = next(model.parameters()).device  # Get model device
    local_scores = []  # Store local prediction scores
    local_labels = []  # Store local true labels

    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
        num_instance = len(data.sources)
        num_batch = math.ceil(num_instance / batch_size)

        for k in range(num_batch):
            # Prepare batch data for evaluation
            # Prepare batch data
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]

            size = len(sources_batch)
            # Sample negative edges for evaluation
            _, negatives_batch = negative_edge_sampler.sample(size)

            # Compute probabilities for positive and negative edges
            pos_prob, neg_prob = model.module.compute_edge_probabilities(
                sources_batch,
                destinations_batch,
                negatives_batch,
                timestamps_batch,
                edge_idxs_batch,
                n_neighbors,
            )

            pos_prob = pos_prob.squeeze().cpu().numpy()
            neg_prob = neg_prob.squeeze().cpu().numpy()

            # Concatenate results and compute metrics for this batch
            y_score = np.concatenate([pos_prob, neg_prob])
            y_true = np.concatenate([
                np.ones_like(pos_prob),
                np.zeros_like(neg_prob)
            ])

            local_scores.append(torch.tensor(y_score, dtype=torch.float, device=device))
            local_labels.append(torch.tensor(y_true, dtype=torch.float, device=device))

    # Concatenate local results for this process
    local_scores = torch.cat(local_scores)
    local_labels = torch.cat(local_labels)

    # Gather results from all processes
    world_size = dist.get_world_size()
    gathered_scores = [None for _ in range(world_size)]
    gathered_labels = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_scores, local_scores.tolist())
    dist.all_gather_object(gathered_labels, local_labels.tolist())

    # Concatenate all gathered results
    y_score = np.concatenate([np.array(x) for x in gathered_scores])
    y_true = np.concatenate([np.array(x) for x in gathered_labels])
    y_pred = (y_score >= 0.5).astype(int)

    # Compute metrics
    val_ap = average_precision_score(y_true, y_score)
    val_auc = roc_auc_score(y_true, y_score)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = (y_pred == y_true).mean()

    return val_ap, val_auc, precision, recall, f1, acc
