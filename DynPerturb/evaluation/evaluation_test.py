import math

import numpy as np
import torch
from model.tgn_0409 import TGN
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)

def eval_edge_prediction_new(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    """
    Evaluate edge prediction with the given model and data.
    """
    # Ensures the random sampler uses a seed for evaluation (i.e., we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)

            # Fetch data for the current batch
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            celltypes_batch = data.celltypes[s_idx:e_idx]  # 读取 celltype 信息
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]

            size = len(sources_batch)
            _, negative_samples = negative_edge_sampler.sample(size)

            # Compute probabilities for positive and negative edges
            pos_prob, neg_prob = model.compute_edge_probabilities(
                sources_batch,
                destinations_batch,
                negative_samples,
                timestamps_batch,
                edge_idxs_batch,
                n_neighbors=n_neighbors,
                celltypes=celltypes_batch  # 传递 celltype 信息
            )

            # Combine positive and negative probabilities
            pred_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            # Compute metrics
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return np.mean(val_ap), np.mean(val_auc)

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import numpy as np
import torch
import math

def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors, num_classes):
    pred_prob = np.zeros((len(data.sources), num_classes))  # Initialize as 2D array (num_samples, num_classes)
    true_labels = np.zeros(len(data.sources))  # Store true labels
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        decoder.eval()
        tgn.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx:e_idx]
            labels_batch = data.labels[s_idx:e_idx]

            # Compute source embeddings
            source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(
                sources_batch, destinations_batch, destinations_batch, timestamps_batch, edge_idxs_batch, n_neighbors)

            # Get predictions (assuming this is a multi-class classification task)
            pred_prob_batch = decoder(source_embedding)
            pred_prob_batch = torch.softmax(pred_prob_batch, dim=-1)

            # Store predictions
            pred_prob[s_idx:e_idx] = pred_prob_batch.cpu().numpy()
            true_labels[s_idx:e_idx] = labels_batch  # Store true labels

    # Compute ROC AUC for each class (one-vs-rest strategy)
    auc_roc = roc_auc_score(true_labels, pred_prob, average=None, multi_class='ovr')

    # Compute precision, recall, and F1 score for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, np.argmax(pred_prob, axis=1), average=None, zero_division=0)

    # Log class-wise metrics
    # for i in range(num_classes):
    #     logger.info(f"Class {i} - AUC: {auc_roc[i]:.4f}, Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}, Support: {support[i]}")

    # return auc_roc, precision, recall, f1, support
    return auc_roc, precision, recall, f1, support, pred_prob

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def eval_edge_prediction_add(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc, val_precision, val_recall, val_f1, val_acc = [], [], [], [], [], []
    
    with torch.no_grad():
        model = model.eval()
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]

            size = len(sources_batch)
            _, negative_samples = negative_edge_sampler.sample(size)

            pos_prob, neg_prob = model.compute_edge_probabilities(
                sources_batch, destinations_batch,
                negative_samples, timestamps_batch,
                edge_idxs_batch, n_neighbors
            )

            y_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            y_true = np.concatenate([np.ones(size), np.zeros(size)])
            y_pred = (y_score >= 0.5).astype(int)  # Binary classification (threshold 0.5)

            # Calculate metrics
            val_ap.append(average_precision_score(y_true, y_score))
            val_auc.append(roc_auc_score(y_true, y_score))
            
            # Compute precision, recall, f1 score
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            val_precision.append(precision)
            val_recall.append(recall)
            val_f1.append(f1)
            
            # Compute accuracy
            accuracy = (y_pred == y_true).mean()
            val_acc.append(accuracy)

    return np.mean(val_ap), np.mean(val_auc), np.mean(val_precision), np.mean(val_recall), np.mean(val_f1), np.mean(val_acc)

