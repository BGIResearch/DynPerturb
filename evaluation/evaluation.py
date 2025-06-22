import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support
)
import torch.distributed as dist
import math


def eval_edge_prediction_add_1(
    model, negative_edge_sampler, data, n_neighbors, batch_size=200
):
    """
    Evaluate edge prediction (binary classification) for a batch of edges.
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
            # Prepare batch data
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]

            size = len(sources_batch)
            # Sample negative edges
            _, negative_samples = negative_edge_sampler.sample(size)

            # Compute probabilities for positive and negative edges
            pos_prob, neg_prob = model.compute_edge_probabilities(
                sources_batch,
                destinations_batch,
                negative_samples,
                timestamps_batch,
                edge_idxs_batch,
                n_neighbors,
            )

            # Concatenate results and compute metrics
            y_score = np.concatenate([
                pos_prob.cpu().numpy(),
                neg_prob.cpu().numpy()
            ])
            y_true = np.concatenate([
                np.ones(size),
                np.zeros(size)
            ])
            y_pred = (y_score >= 0.5).astype(int)

            val_ap.append(average_precision_score(y_true, y_score))
            val_auc.append(roc_auc_score(y_true, y_score))
            val_acc.append(accuracy_score(y_true, y_pred))
            val_f1.append(f1_score(y_true, y_pred))

    return (
        np.mean(val_ap),
        np.mean(val_auc),
        np.mean(val_acc),
        np.mean(val_f1)
    )


def eval_node_classification_ddp_new(
    netmodel, data, edge_idxs, batch_size, n_neighbors, num_classes
):
    """
    Evaluate node classification in distributed setting.
    Returns per-class AUC, precision, recall, F1, support, predicted
    probabilities, and true labels.
    """
    # device = next(netmodel.parameters()).device  # Get model device
    local_pred_prob = []  # Store local predicted probabilities
    local_true_labels = []  # Store local true labels

    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        netmodel.eval()  # Set model to evaluation mode

        for k in range(num_batch):
            # Prepare batch data
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

            logits = netmodel.module.node_classification_decoder(
                source_embedding
            )
            prob = torch.softmax(logits, dim=-1)

            local_pred_prob.append(prob.cpu())
            local_true_labels.append(
                torch.tensor(labels_batch, dtype=torch.long)
            )

    # Gather predictions and labels from all processes
    local_pred_prob = torch.cat(local_pred_prob, dim=0).cpu().numpy()
    local_true_labels = torch.cat(local_true_labels, dim=0).cpu().numpy()

    gathered_probs = [None] * dist.get_world_size()
    gathered_labels = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_probs, local_pred_prob.tolist())
    dist.all_gather_object(gathered_labels, local_true_labels.tolist())

    all_pred_prob = np.concatenate([
        np.array(p) for p in gathered_probs
    ], axis=0)
    all_true_labels = np.concatenate([
        np.array(label) for label in gathered_labels
    ], axis=0)

    try:
        # Compute per-class and weighted AUC
        auc_roc = roc_auc_score(
            all_true_labels, all_pred_prob, average=None, multi_class="ovr"
        )
        # weighted_auc_roc = roc_auc_score(
        #     all_true_labels, all_pred_prob, average="weighted", multi_class="ovr"
        # )
    except ValueError as e:
        print(f"[Warning] AUC error: {e}")
        auc_roc = np.full(num_classes, np.nan)

    # Compute precision, recall, F1, support
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true_labels,
        np.argmax(all_pred_prob, axis=1),
        average=None,
        zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, weighted_support = (
        precision_recall_fscore_support(
            all_true_labels,
            np.argmax(all_pred_prob, axis=1),
            average="weighted",
            zero_division=0,
        )
    )

    return (
        auc_roc,
        precision,
        recall,
        f1,
        support,
        all_pred_prob,
        all_true_labels
    )


def eval_edge_prediction_add_ddp(
    model, negative_edge_sampler, data, n_neighbors, batch_size=200
):
    """
    Evaluate edge prediction in distributed setting. Returns AP, AUC, precision, recall, F1, accuracy.
    """
    assert negative_edge_sampler.seed is not None  # Ensure sampler is seeded
    negative_edge_sampler.reset_random_state()  # Reset sampler state

    val_ap, val_auc, val_precision, val_recall, val_f1, val_acc = (
        [], [], [], [], [], []
    )

    with torch.no_grad():
        model = model.eval()  # Set model to evaluation mode
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            # Prepare batch data
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]

            size = len(sources_batch)
            # Sample negative edges
            _, negative_samples = negative_edge_sampler.sample(size)

            # Compute probabilities for positive and negative edges
            pos_prob, neg_prob = model.module.compute_edge_probabilities(
                sources_batch,
                destinations_batch,
                negative_samples,
                timestamps_batch,
                edge_idxs_batch,
                n_neighbors,
            )

            # Concatenate results and compute metrics
            y_score = np.concatenate([
                pos_prob.cpu().numpy(),
                neg_prob.cpu().numpy()
            ])
            y_true = np.concatenate([
                np.ones(size),
                np.zeros(size)
            ])
            y_pred = (y_score >= 0.5).astype(int)

            val_ap.append(average_precision_score(y_true, y_score))
            val_auc.append(roc_auc_score(y_true, y_score))

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            val_precision.append(precision)
            val_recall.append(recall)
            val_f1.append(f1)

            accuracy = (y_pred == y_true).mean()
            val_acc.append(accuracy)

    return (
        np.mean(val_ap),
        np.mean(val_auc),
        np.mean(val_precision),
        np.mean(val_recall),
        np.mean(val_f1),
        np.mean(val_acc)
    )
