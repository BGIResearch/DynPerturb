import math

import numpy as np
import torch
from model.tgn_mulit import TGN
from sklearn.metrics import average_precision_score, roc_auc_score

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def eval_node_classification(tgn, data, edge_idxs, batch_size, n_neighbors, num_classes):
    pred_prob = np.zeros((len(data.sources), num_classes))  # Initialize as 2D array (num_samples, num_classes)
    true_labels = np.zeros(len(data.sources))  # Store true labels
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        tgn.eval()  # Ensure model is in evaluation mode
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx:e_idx]
            labels_batch = data.labels[s_idx:e_idx]

            # 计算源节点的嵌入
            source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(
                sources_batch, destinations_batch, destinations_batch, timestamps_batch, edge_idxs_batch, n_neighbors
            )

            # 直接使用模型的输出（而不是外部的解码器）
            node_classification_logits = tgn.node_classification_decoder(source_embedding)

            # 使用 softmax 获得类别概率
            pred_prob_batch = torch.softmax(node_classification_logits, dim=-1)
            

            # 存储预测结果
            pred_prob[s_idx:e_idx] = pred_prob_batch.cpu().numpy()
            true_labels[s_idx:e_idx] = labels_batch  # 存储真实标签

    # 计算每个类的 AUC 值（one-vs-rest 策略）
    auc_roc = roc_auc_score(true_labels, pred_prob, average=None, multi_class='ovr')

    # 计算每个类的精度、召回率和 F1 分数
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, np.argmax(pred_prob, axis=1), average=None, zero_division=0)

    # 返回评估结果
    return auc_roc, precision, recall, f1, support, pred_prob


def eval_node_classification_ddp(tgn, data, edge_idxs, batch_size, n_neighbors, num_classes):
    pred_prob = np.zeros((len(data.sources), num_classes))  # Initialize as 2D array (num_samples, num_classes)
    true_labels = np.zeros(len(data.sources))  # Store true labels
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        tgn.eval()  # Ensure model is in evaluation mode
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx:e_idx]
            labels_batch = data.labels[s_idx:e_idx]

            # 计算源节点的嵌入
            source_embedding, destination_embedding, _ = tgn.module.compute_temporal_embeddings(
                sources_batch, destinations_batch, destinations_batch, timestamps_batch, edge_idxs_batch, n_neighbors
            )

            # 直接使用模型的输出（而不是外部的解码器）
            node_classification_logits = tgn.module.node_classification_decoder(source_embedding)

            # 使用 softmax 获得类别概率
            pred_prob_batch = torch.softmax(node_classification_logits, dim=-1)
            

            # 存储预测结果
            pred_prob[s_idx:e_idx] = pred_prob_batch.cpu().numpy()
            true_labels[s_idx:e_idx] = labels_batch  # 存储真实标签

    # 计算每个类的 AUC 值（one-vs-rest 策略）
    auc_roc = roc_auc_score(true_labels, pred_prob, average=None, multi_class='ovr')

    # 计算每个类的精度、召回率和 F1 分数
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, np.argmax(pred_prob, axis=1), average=None, zero_division=0)

    # 返回评估结果
    return auc_roc, precision, recall, f1, support, pred_prob




from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

def eval_edge_prediction_add(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc, val_precision, val_recall, val_f1, val_acc = [], [], [], [], [], []

    with torch.no_grad():
        model = model.eval()  # Ensure model is in evaluation mode
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

            # 直接计算正负边的概率
            pos_prob, neg_prob = model.compute_edge_probabilities(
                sources_batch, destinations_batch,
                negative_samples, timestamps_batch,
                edge_idxs_batch, n_neighbors
            )

            # 合并正负样本的预测分数
            y_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            y_true = np.concatenate([np.ones(size), np.zeros(size)])
            y_pred = (y_score >= 0.5).astype(int)  # 二分类（阈值为 0.5）

            # 计算评价指标
            val_ap.append(average_precision_score(y_true, y_score))
            val_auc.append(roc_auc_score(y_true, y_score))

            # 计算精度、召回率、F1分数
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            val_precision.append(precision)
            val_recall.append(recall)
            val_f1.append(f1)

            # 计算准确率
            accuracy = (y_pred == y_true).mean()
            val_acc.append(accuracy)

    # 返回平均值
    return np.mean(val_ap), np.mean(val_auc), np.mean(val_precision), np.mean(val_recall), np.mean(val_f1), np.mean(val_acc)

def eval_edge_prediction_add_ddp(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc, val_precision, val_recall, val_f1, val_acc = [], [], [], [], [], []

    with torch.no_grad():
        model = model.eval()  # Ensure model is in evaluation mode
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

            # 直接计算正负边的概率
            pos_prob, neg_prob = model.module.compute_edge_probabilities(
                sources_batch, destinations_batch,
                negative_samples, timestamps_batch,
                edge_idxs_batch, n_neighbors
            )

            # 合并正负样本的预测分数
            y_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            y_true = np.concatenate([np.ones(size), np.zeros(size)])
            y_pred = (y_score >= 0.5).astype(int)  # 二分类（阈值为 0.5）

            # 计算评价指标
            val_ap.append(average_precision_score(y_true, y_score))
            val_auc.append(roc_auc_score(y_true, y_score))

            # 计算精度、召回率、F1分数
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            val_precision.append(precision)
            val_recall.append(recall)
            val_f1.append(f1)

            # 计算准确率
            accuracy = (y_pred == y_true).mean()
            val_acc.append(accuracy)

    # 返回平均值
    return np.mean(val_ap), np.mean(val_auc), np.mean(val_precision), np.mean(val_recall), np.mean(val_f1), np.mean(val_acc)


import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import torch.distributed as dist
import math

def eval_node_classification_ddp_new(tgn, data, edge_idxs, batch_size, n_neighbors, num_classes):
    device = next(tgn.parameters()).device
    local_pred_prob = []
    local_true_labels = []

    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        tgn.eval()

        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx:e_idx]
            labels_batch = data.labels[s_idx:e_idx]

            source_embedding, destination_embedding, _ = tgn.module.compute_temporal_embeddings(
                sources_batch, destinations_batch, destinations_batch,
                timestamps_batch, edge_idxs_batch, n_neighbors
            )

            logits = tgn.module.node_classification_decoder(source_embedding)
            prob = torch.softmax(logits, dim=-1)

            local_pred_prob.append(prob.cpu())
            local_true_labels.append(torch.tensor(labels_batch, dtype=torch.long))

    # concat 当前进程
    local_pred_prob = torch.cat(local_pred_prob, dim=0).cpu().numpy()
    local_true_labels = torch.cat(local_true_labels, dim=0).cpu().numpy()

    # all_gather_object 更兼容变长 batch
    gathered_probs, gathered_labels = [None] * dist.get_world_size(), [None] * dist.get_world_size()
    dist.all_gather_object(gathered_probs, local_pred_prob.tolist())
    dist.all_gather_object(gathered_labels, local_true_labels.tolist())

    # 拼接回完整样本
    all_pred_prob = np.concatenate([np.array(p) for p in gathered_probs], axis=0)
    all_true_labels = np.concatenate([np.array(l) for l in gathered_labels], axis=0)

    # 防止某些类缺失时 AUC 报错
    try:
        auc_roc = roc_auc_score(all_true_labels, all_pred_prob, average=None, multi_class='ovr')
        weighted_auc_roc = roc_auc_score(all_true_labels, all_pred_prob, average='weighted', multi_class='ovr')

    except ValueError as e:
        print(f"[Warning] AUC error: {e}")
        auc_roc = np.full(num_classes, np.nan)

    precision, recall, f1, support = precision_recall_fscore_support(
        all_true_labels, np.argmax(all_pred_prob, axis=1), average=None, zero_division=0
    )
    # Optionally, calculate weighted average metrics (useful in class imbalance cases)
    weighted_precision, weighted_recall, weighted_f1, weighted_support = precision_recall_fscore_support(
        all_true_labels, np.argmax(all_pred_prob, axis=1), average='weighted', zero_division=0
    )

    return auc_roc, precision, recall, f1, support, all_pred_prob,all_true_labels

import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
import torch.distributed as dist
import math

def eval_edge_prediction_add_ddp_new(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    device = next(model.parameters()).device
    local_scores = []
    local_labels = []

    with torch.no_grad():
        model.eval()
        num_instance = len(data.sources)
        num_batch = math.ceil(num_instance / batch_size)

        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]

            size = len(sources_batch)
            _, negatives_batch = negative_edge_sampler.sample(size)

            pos_prob, neg_prob = model.module.compute_edge_probabilities(
                sources_batch, destinations_batch, negatives_batch,
                timestamps_batch, edge_idxs_batch, n_neighbors
            )

            pos_prob = pos_prob.squeeze().cpu().numpy()
            neg_prob = neg_prob.squeeze().cpu().numpy()

            y_score = np.concatenate([pos_prob, neg_prob])
            y_true = np.concatenate([np.ones_like(pos_prob), np.zeros_like(neg_prob)])

            local_scores.append(torch.tensor(y_score, dtype=torch.float, device=device))
            local_labels.append(torch.tensor(y_true, dtype=torch.float, device=device))

    # concat 当前进程
    local_scores = torch.cat(local_scores)
    local_labels = torch.cat(local_labels)

    # gather
    world_size = dist.get_world_size()
    gathered_scores = [None for _ in range(world_size)]
    gathered_labels = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_scores, local_scores.tolist())
    dist.all_gather_object(gathered_labels, local_labels.tolist())

    y_score = np.concatenate([np.array(x) for x in gathered_scores])
    y_true = np.concatenate([np.array(x) for x in gathered_labels])
    y_pred = (y_score >= 0.5).astype(int)

    val_ap = average_precision_score(y_true, y_score)
    val_auc = roc_auc_score(y_true, y_score)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    acc = (y_pred == y_true).mean()

    return val_ap, val_auc, precision, recall, f1, acc

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import torch.distributed as dist
import math

def eval_node_classification_ddp_new1(tgn, data, edge_idxs, batch_size, n_neighbors, num_classes):
    device = next(tgn.parameters()).device
    local_pred_prob = []
    local_true_labels = []

    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        tgn.eval()

        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx:e_idx]
            labels_batch = data.labels[s_idx:e_idx]

            # Get node embeddings
            source_embedding, destination_embedding, _ = tgn.module.compute_temporal_embeddings(
                sources_batch, destinations_batch, destinations_batch,
                timestamps_batch, edge_idxs_batch, n_neighbors
            )

            # Compute logits and probabilities
            logits = tgn.module.node_classification_decoder(source_embedding)
            prob = torch.softmax(logits, dim=-1)

            # Store predictions and true labels
            local_pred_prob.append(prob.cpu())
            local_true_labels.append(torch.tensor(labels_batch, dtype=torch.long))

    # Concatenate predictions and labels across all processes
    local_pred_prob = torch.cat(local_pred_prob, dim=0).cpu().numpy()
    local_true_labels = torch.cat(local_true_labels, dim=0).cpu().numpy()

    # Gather predictions and labels across all processes
    gathered_probs, gathered_labels = [None] * dist.get_world_size(), [None] * dist.get_world_size()
    dist.all_gather_object(gathered_probs, local_pred_prob.tolist())
    dist.all_gather_object(gathered_labels, local_true_labels.tolist())

    # Concatenate to form the complete dataset
    all_pred_prob = np.concatenate([np.array(p) for p in gathered_probs], axis=0)
    all_true_labels = np.concatenate([np.array(l) for l in gathered_labels], axis=0)

    # Compute weighted AUC
    try:
        auc_roc = roc_auc_score(all_true_labels, all_pred_prob, average='weighted', multi_class='ovr')
    except ValueError as e:
        print(f"[Warning] AUC error: {e}")
        auc_roc = np.full(num_classes, np.nan)

    # Compute Precision, Recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true_labels, np.argmax(all_pred_prob, axis=1), average=None, zero_division=0
    )

    # Compute weighted average F1
    f1_weighted = np.average(f1, weights=support)

    # Compute macro and micro average metrics
    precision_macro, recall_macro, f1_macro, support_macro = precision_recall_fscore_support(
        all_true_labels, np.argmax(all_pred_prob, axis=1), average='macro', zero_division=0
    )

    precision_micro, recall_micro, f1_micro, support_micro = precision_recall_fscore_support(
        all_true_labels, np.argmax(all_pred_prob, axis=1), average='micro', zero_division=0
    )

    return auc_roc, precision, recall, f1, support, f1_weighted, precision_macro, recall_macro, f1_macro, support_macro, precision_micro, recall_micro, f1_micro, support_micro, all_pred_prob


import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
import torch.distributed as dist
import math

def eval_edge_prediction_add_ddp_new1(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    device = next(model.parameters()).device
    local_scores = []
    local_labels = []

    with torch.no_grad():
        model.eval()
        num_instance = len(data.sources)
        num_batch = math.ceil(num_instance / batch_size)

        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]

            size = len(sources_batch)
            _, negatives_batch = negative_edge_sampler.sample(size)

            # 计算连接预测的正负样本概率
            pos_prob, neg_prob = model.module.compute_edge_probabilities(
                sources_batch, destinations_batch, negatives_batch,
                timestamps_batch, edge_idxs_batch, n_neighbors
            )

            pos_prob = pos_prob.squeeze().cpu().numpy()
            neg_prob = neg_prob.squeeze().cpu().numpy()

            y_score = np.concatenate([pos_prob, neg_prob])
            y_true = np.concatenate([np.ones_like(pos_prob), np.zeros_like(neg_prob)])

            # 记录每个进程的 scores 和 labels
            local_scores.append(torch.tensor(y_score, dtype=torch.float, device=device))
            local_labels.append(torch.tensor(y_true, dtype=torch.float, device=device))

    # 合并所有进程的结果
    local_scores = torch.cat(local_scores)
    local_labels = torch.cat(local_labels)

    # 收集所有进程的数据
    world_size = dist.get_world_size()
    gathered_scores = [None for _ in range(world_size)]
    gathered_labels = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_scores, local_scores.tolist())
    dist.all_gather_object(gathered_labels, local_labels.tolist())

    # 将所有进程的结果拼接成完整的预测数据
    y_score = np.concatenate([np.array(x) for x in gathered_scores])
    y_true = np.concatenate([np.array(x) for x in gathered_labels])
    
    # 将预测分数转为预测标签
    y_pred = (y_score >= 0.5).astype(int)

    # 计算 AUC 和其他评估指标
    val_ap = average_precision_score(y_true, y_score)
    val_auc = roc_auc_score(y_true, y_score, average='weighted')  # 加权 AUC
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    acc = (y_pred == y_true).mean()

    return val_ap, val_auc, precision, recall, f1, acc

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch

from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, hamming_loss
)
import torch.distributed as dist
import numpy as np
import torch

from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, hamming_loss
)
import torch.distributed as dist
import numpy as np
import torch
import math

def eval_node_classification_ddp_label(tgn, data, edge_idxs, batch_size, n_neighbors, num_classes, threshold=0.5):
    device = next(tgn.parameters()).device
    local_pred_prob = []
    local_true_labels = []

    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        tgn.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx:e_idx]
            labels_batch = data.labels[s_idx:e_idx]

            source_embedding, _, _ = tgn.module.compute_temporal_embeddings(
                sources_batch, destinations_batch, destinations_batch,
                timestamps_batch, edge_idxs_batch, n_neighbors
            )

            logits = tgn.module.node_classification_decoder(source_embedding)  # shape: [B, num_classes]
            probs = torch.sigmoid(logits).cpu().numpy()  # 多标签分类用 sigmoid
            local_pred_prob.append(probs)
            local_true_labels.append(labels_batch)

    # 拼接本地 batch 结果
    local_pred_prob = np.concatenate(local_pred_prob, axis=0)
    local_true_labels = np.concatenate(local_true_labels, axis=0)

    # DDP 多进程同步
    world_size = dist.get_world_size()
    local_pred_prob_tensor = torch.tensor(local_pred_prob, device=device)
    local_true_labels_tensor = torch.tensor(local_true_labels, device=device)

    gathered_preds = [torch.zeros_like(local_pred_prob_tensor) for _ in range(world_size)]
    gathered_trues = [torch.zeros_like(local_true_labels_tensor) for _ in range(world_size)]

    dist.all_gather(gathered_preds, local_pred_prob_tensor)
    dist.all_gather(gathered_trues, local_true_labels_tensor)

    all_pred_prob = torch.cat(gathered_preds, dim=0).cpu().numpy()
    all_true_labels = torch.cat(gathered_trues, dim=0).cpu().numpy()

    y_pred = (all_pred_prob >= threshold).astype(int)
    y_true = all_true_labels.astype(int)

    # 评估指标
    try:
        auc_per_class = roc_auc_score(y_true, all_pred_prob, average=None)
        auc_macro = roc_auc_score(y_true, all_pred_prob, average='macro')
    except ValueError as e:
        print(f"[Warning] AUC error: {e}")
        auc_per_class = np.full(num_classes, np.nan)
        auc_macro = np.nan

    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    support_per_class = y_true.sum(axis=0)

    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    pr_auc_macro = average_precision_score(y_true, all_pred_prob, average='macro')
    hamming = hamming_loss(y_true, y_pred)

    return auc_per_class, precision_per_class, recall_per_class, f1_per_class, support_per_class, all_pred_prob, f1_macro, f1_micro, precision_macro, recall_macro, auc_macro, pr_auc_macro, hamming
