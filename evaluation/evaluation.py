
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# ---------- 用于 kidney 的非DDP边预测评估 ----------
def eval_edge_prediction_add(netmodel, decoder, pos_data, neg_data, args):
    netmodel.eval()
    decoder.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for (u, i, ts, eidx) in pos_data:
            u, i, ts, eidx = u.to(args.device), i.to(args.device), ts.to(args.device), eidx.to(args.device)
            emb_u, _ = netmodel(u, ts, eidx, n_neighbors=args.n_neighbor)
            emb_i, _ = netmodel(i, ts, eidx, n_neighbors=args.n_neighbor)
            score = decoder(emb_u, emb_i).squeeze()
            y_true.extend([1] * len(score))
            y_score.extend(score.cpu().numpy())
        for (u, i, ts, eidx) in neg_data:
            u, i, ts, eidx = u.to(args.device), i.to(args.device), ts.to(args.device), eidx.to(args.device)
            emb_u, _ = netmodel(u, ts, eidx, n_neighbors=args.n_neighbor)
            emb_i, _ = netmodel(i, ts, eidx, n_neighbors=args.n_neighbor)
            score = decoder(emb_u, emb_i).squeeze()
            y_true.extend([0] * len(score))
            y_score.extend(score.cpu().numpy())
    y_pred = (np.array(y_score) > 0.5).astype(int)
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    return auc, ap, f1

# ---------- 用于 DDP 的边预测评估 ----------
def eval_edge_prediction_add_ddp_new1(netmodel, decoder, pos_data, neg_data, args):
    netmodel.eval()
    decoder.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for (u, i, ts, eidx) in pos_data:
            u, i, ts, eidx = u.to(args.device), i.to(args.device), ts.to(args.device), eidx.to(args.device)
            emb_u, _ = netmodel(u, ts, eidx, n_neighbors=args.n_neighbor)
            emb_i, _ = netmodel(i, ts, eidx, n_neighbors=args.n_neighbor)
            score = decoder(emb_u, emb_i).squeeze()
            y_true.extend([1] * len(score))
            y_score.extend(score.cpu().numpy())
        for (u, i, ts, eidx) in neg_data:
            u, i, ts, eidx = u.to(args.device), i.to(args.device), ts.to(args.device), eidx.to(args.device)
            emb_u, _ = netmodel(u, ts, eidx, n_neighbors=args.n_neighbor)
            emb_i, _ = netmodel(i, ts, eidx, n_neighbors=args.n_neighbor)
            score = decoder(emb_u, emb_i).squeeze()
            y_true.extend([0] * len(score))
            y_score.extend(score.cpu().numpy())
    y_pred = (np.array(y_score) > 0.5).astype(int)
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    return auc, ap, f1

# ---------- 用于 DDP 的节点分类 ----------

def eval_node_classification_ddp_new1(netmodel, decoder, labels, val_data, args):
    netmodel.eval()
    decoder.eval()
    val_label, val_pred = [], []
    with torch.no_grad():
        for batch in val_data:
            node_ids, timestamps, edge_idxs, labels_batch = batch
            node_ids = node_ids.to(args.device)
            timestamps = timestamps.to(args.device)
            edge_idxs = edge_idxs.to(args.device)
            labels_batch = labels_batch.to(args.device)
            embeddings, _ = netmodel(node_ids, timestamps, edge_idxs, n_neighbors=args.n_neighbor)
            pred = decoder(embeddings).squeeze()
            val_label.extend(labels_batch.cpu().numpy())
            val_pred.extend(pred.cpu().numpy())
    pred_label = (np.array(val_pred) > 0.5).astype(int)
    f1_macro = f1_score(val_label, pred_label, average='macro')
    f1_micro = f1_score(val_label, pred_label, average='micro')
    return f1_macro, f1_micro

# ---------- 多标签分类（sigmoid） ----------

def eval_node_classification_ddp_label(netmodel, decoder, labels, val_data, args):
    netmodel.eval()
    decoder.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_data:
            node_ids, timestamps, edge_idxs, labels_batch = batch
            node_ids = node_ids.to(args.device)
            timestamps = timestamps.to(args.device)
            edge_idxs = edge_idxs.to(args.device)
            labels_batch = labels_batch.to(args.device)
            emb, _ = netmodel(node_ids, timestamps, edge_idxs, n_neighbors=args.n_neighbor)
            pred = decoder(emb).sigmoid().cpu().numpy()
            y_pred.append(pred)
            y_true.append(labels_batch.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_pred_label = (y_pred > 0.5).astype(int)
    f1_micro = f1_score(y_true, y_pred_label, average='micro')
    f1_macro = f1_score(y_true, y_pred_label, average='macro')
    return f1_macro, f1_micro

