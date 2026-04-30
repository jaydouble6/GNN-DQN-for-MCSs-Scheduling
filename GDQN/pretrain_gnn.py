import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt
import copy
import math
import os
from torch.utils.data import Subset
from env.config import *

from GDQN.net import MCSHeteroGNN, IEVHeteroGNN


def _get_node_feat_dim(graph_data, node_type):
    if node_type not in graph_data.node_types:
        return None
    x = graph_data[node_type].x
    if x is None or x.dim() != 2:
        return None
    return int(x.size(1))


def _validate_dataset_feature_dims(mcs_dataset, iev_dataset):
    if len(mcs_dataset) == 0 or len(iev_dataset) == 0:
        return

    mcs_sample = mcs_dataset[0]
    iev_sample = iev_dataset[0]

    expected_mcs = {
        "idle": MCS_FEAT_DIM,
        "task": MCS_FEAT_DIM,
        "quasi": VEHICLE_FEAT_DIM,
    }
    expected_iev = {
        "iev": VEHICLE_FEAT_DIM,
        "task": MCS_FEAT_DIM,
    }

    mismatches = []
    for node_type, expected_dim in expected_mcs.items():
        actual_dim = _get_node_feat_dim(mcs_sample, node_type)
        if actual_dim is not None and actual_dim != expected_dim:
            mismatches.append(f"MCS图节点 `{node_type}`: 数据维度={actual_dim}, 配置维度={expected_dim}")

    for node_type, expected_dim in expected_iev.items():
        actual_dim = _get_node_feat_dim(iev_sample, node_type)
        if actual_dim is not None and actual_dim != expected_dim:
            mismatches.append(f"IEV图节点 `{node_type}`: 数据维度={actual_dim}, 配置维度={expected_dim}")

    if mismatches:
        mismatch_text = "\n".join([f"  - {m}" for m in mismatches])
        raise RuntimeError(
            "检测到离线图数据特征维度与当前配置不一致，无法进行预训练。\n"
            f"{mismatch_text}\n"
            "请重新运行 GDQN/collect_data.py 生成与当前 config 一致的数据集。"
        )


def mask_edges_symmetrically(batch_data, drop_rate=0.2, edge_types_to_mask=None):
    train_data = copy.copy(batch_data)
    train_data.edge_index_dict = {}
    supervision_dict = {}

    if edge_types_to_mask is None:
        edge_types_to_mask = batch_data.edge_types

    cross_pairs = []
    seen_cross_pairs = set()
    peer_types = []

    for etype in edge_types_to_mask:
        src, rel, dst = etype
        if src != dst:
            reverse_etype = (dst, rel, src)
            if reverse_etype in edge_types_to_mask:
                # 无序对去重，避免 (A,B) 与 (B,A) 被重复加入。
                pair_key = tuple(sorted([etype, reverse_etype]))
                if pair_key not in seen_cross_pairs:
                    seen_cross_pairs.add(pair_key)
                    cross_pairs.append((etype, reverse_etype))
        else:
            peer_types.append(etype)

    for e_forward, e_reverse in cross_pairs:
        if e_forward in batch_data.edge_types:
            edge_index = batch_data[e_forward].edge_index
            num_edges = edge_index.size(1)

            if num_edges == 0:
                continue

            perm = torch.randperm(num_edges)
            num_drop = int(num_edges * drop_rate)
            num_drop = max(1, num_drop) if num_edges > 1 else 0

            drop_idx = perm[:num_drop]
            keep_idx = perm[num_drop:]

            train_data[e_forward].edge_index = edge_index[:, keep_idx]
            if e_reverse in batch_data.edge_types:
                train_data[e_reverse].edge_index = batch_data[e_reverse].edge_index[:, keep_idx]

            supervision_dict[e_forward] = edge_index[:, drop_idx]

    for peer_type in peer_types:
        if peer_type in batch_data.edge_types:
            edge_index = batch_data[peer_type].edge_index
            num_edges = edge_index.size(1)

            if num_edges == 0:
                continue

            # 优先按“成对双向边”处理；若边数异常（如奇数）则退化到普通随机掩码。
            if num_edges >= 2 and num_edges % 2 == 0:
                num_pairs = num_edges // 2
                perm = torch.randperm(num_pairs)
                num_drop_pairs = int(num_pairs * drop_rate)
                num_drop_pairs = max(1, num_drop_pairs) if num_pairs > 1 else 0

                drop_pairs = perm[:num_drop_pairs]
                keep_pairs = perm[num_drop_pairs:]

                keep_idx = torch.cat([keep_pairs * 2, keep_pairs * 2 + 1]) if len(keep_pairs) > 0 else torch.empty(
                    0, dtype=torch.long)
                drop_idx = torch.cat([drop_pairs * 2, drop_pairs * 2 + 1]) if len(drop_pairs) > 0 else torch.empty(
                    0, dtype=torch.long)
            else:
                perm = torch.randperm(num_edges)
                num_drop = int(num_edges * drop_rate)
                num_drop = max(1, num_drop) if num_edges > 1 else 0
                drop_idx = perm[:num_drop]
                keep_idx = perm[num_drop:]

            train_data[peer_type].edge_index = edge_index[:, keep_idx]
            supervision_dict[peer_type] = edge_index[:, drop_idx]

    for etype in batch_data.edge_types:
        if etype not in train_data.edge_types:
            train_data[etype].edge_index = batch_data[etype].edge_index

    return train_data, supervision_dict


def train_single_gnn(dataset, model_class, model_name, hidden_channels=32, epochs=100, val_ratio=0.2, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 当前使用的计算设备: {device}")

    batch_size = 128
    dataset_size = len(dataset)
    if dataset_size == 0:
        print(f"❌ {model_name} 数据集为空，无法训练。")
        return None

    if dataset_size == 1:
        train_dataset = dataset
        val_dataset = []
    else:
        val_size = int(dataset_size * val_ratio)
        val_size = max(1, val_size)
        if val_size >= dataset_size:
            val_size = dataset_size - 1
        train_size = dataset_size - val_size

        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(dataset_size, generator=generator).tolist()
        train_idx = perm[:train_size]
        val_idx = perm[train_size:]
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_dataset) > 0 else None
    print(f"📦 {model_name} 数据集总图数: {dataset_size} | 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")
    print(f"📦 {model_name} 训练 Batch 数量: {len(train_loader)} | 验证 Batch 数量: {len(val_loader) if val_loader else 0}")

    gnn_model = model_class(hidden_channels=hidden_channels, dropout=0.1, heads=2, num_layers=2).to(device)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scale_factor = math.sqrt(hidden_channels)

    def edge_group_key(etype):
        src, rel, dst = etype
        if src == dst:
            return src, rel, dst
        a, b = sorted([src, dst])
        return a, rel, b

    def edge_group_display(group_key):
        src, rel, dst = group_key
        return f"{src:<5} <-> {rel:<4} <-> {dst:<5}"

    edge_loss_weights = {etype: 1.0 for etype in dataset[0].edge_types}
    edge_groups = []
    for etype in edge_loss_weights.keys():
        g_key = edge_group_key(etype)
        if g_key not in edge_groups:
            edge_groups.append(g_key)

    print(f"🚀 开始预训练 {model_name} GNN...")

    history_global_loss = []
    history_val_global_loss = []
    history_edge_loss = {k: [] for k in edge_groups}

    def run_one_epoch(data_loader, model, optimizer_ref=None):
        is_train = optimizer_ref is not None
        if is_train:
            model.train()
        else:
            model.eval()

        total_loss_local = 0.0
        valid_batches_local = 0
        total_batches_local = len(data_loader)
        edge_loss_dict_local = {k: 0.0 for k in edge_groups}
        edge_count_dict_local = {k: 0 for k in edge_groups}

        for batch_data in data_loader:
            train_batch, supervision_edges = mask_edges_symmetrically(batch_data, drop_rate=0.2)
            if len(supervision_edges) == 0:
                continue

            train_batch = train_batch.to(device)
            if is_train:
                optimizer_ref.zero_grad()

            with torch.set_grad_enabled(is_train):
                h_dict = model(train_batch)
                batch_loss = 0
                valid_types_in_batch = 0

                for edge_type, pos_edge_index in supervision_edges.items():
                    if pos_edge_index.size(1) == 0:
                        continue

                    src_type, _, dst_type = edge_type
                    if src_type not in h_dict or dst_type not in h_dict:
                        continue

                    num_src_nodes = h_dict[src_type].size(0)
                    num_dst_nodes = h_dict[dst_type].size(0)
                    if num_src_nodes == 0 or num_dst_nodes == 0:
                        continue

                    valid_types_in_batch += 1
                    pos_edge_index = pos_edge_index.to(device)

                    src_pos = h_dict[src_type][pos_edge_index[0]]
                    dst_pos = h_dict[dst_type][pos_edge_index[1]]
                    pos_pred = (src_pos * dst_pos).sum(dim=-1) / scale_factor

                    neg_edge_index = negative_sampling(
                        edge_index=pos_edge_index,
                        num_nodes=(num_src_nodes, num_dst_nodes),
                        num_neg_samples=pos_edge_index.size(1)
                    ).to(device)

                    src_neg = h_dict[src_type][neg_edge_index[0]]
                    dst_neg = h_dict[dst_type][neg_edge_index[1]]
                    neg_pred = (src_neg * dst_neg).sum(dim=-1) / scale_factor

                    preds = torch.cat([pos_pred, neg_pred])
                    labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

                    edge_loss = F.binary_cross_entropy_with_logits(preds, labels)
                    g_key = edge_group_key(edge_type)
                    if g_key in edge_loss_dict_local:
                        edge_loss_dict_local[g_key] += edge_loss.item()
                        edge_count_dict_local[g_key] += 1

                    weight = edge_loss_weights.get(edge_type, 1.0)
                    batch_loss += (edge_loss * weight)

                if valid_types_in_batch > 0:
                    batch_loss = batch_loss / valid_types_in_batch
                    if is_train:
                        batch_loss.backward()
                        optimizer_ref.step()

                    total_loss_local += batch_loss.item()
                    valid_batches_local += 1

        avg_loss_local = total_loss_local / valid_batches_local if valid_batches_local > 0 else 0
        return avg_loss_local, edge_loss_dict_local, edge_count_dict_local, total_batches_local, valid_batches_local

    for epoch in range(epochs):
        avg_loss, edge_loss_dict, edge_count_dict, total_batches, valid_batches = run_one_epoch(
            train_loader, gnn_model, optimizer_ref=optimizer
        )
        if val_loader is not None:
            val_loss, _, _, _, _ = run_one_epoch(val_loader, gnn_model, optimizer_ref=None)
        else:
            val_loss = None

        if val_loss is None:
            print(f"\nEpoch {epoch + 1:03d}/{epochs} | {model_name} Train Global Loss: {avg_loss:.4f}")
        else:
            print(
                f"\nEpoch {epoch + 1:03d}/{epochs} | {model_name} Train/Val Global Loss: "
                f"{avg_loss:.4f}/{val_loss:.4f}"
            )

        for g_key in edge_groups:
            if edge_count_dict[g_key] > 0:
                avg_etype_loss = edge_loss_dict[g_key] / edge_count_dict[g_key]
                coverage = edge_count_dict[g_key] / max(1, total_batches)
                print(
                    f"    ├─ {edge_group_display(g_key)} | Raw Loss: {avg_etype_loss:.4f} "
                    f"| 覆盖率: {coverage * 100:.1f}% ({edge_count_dict[g_key]}/{total_batches})"
                )
            else:
                print(
                    f"    ├─ {edge_group_display(g_key)} | Raw Loss: N/A (未出现) "
                    f"| 覆盖率: 0.0% (0/{total_batches})"
                )
        print(f"    └─ 有效训练 Batch: {valid_batches}/{total_batches}")

        scheduler.step(avg_loss)

        history_global_loss.append(avg_loss)
        history_val_global_loss.append(val_loss)
        for g_key in edge_groups:
            if edge_count_dict[g_key] > 0:
                history_edge_loss[g_key].append(edge_loss_dict[g_key] / edge_count_dict[g_key])
            else:
                history_edge_loss[g_key].append(None)

    os.makedirs('./models', exist_ok=True)
    torch.save(gnn_model.state_dict(), f'./models/pretrained_{model_name.lower()}_gnn.pth')
    print(f"\n✅ {model_name} GNN 预训练完成！权重已保存至 ./models/pretrained_{model_name.lower()}_gnn.pth")

    print(f"📊 正在生成 {model_name} Loss 折线图...")
    plt.figure(figsize=(12, 7))

    epoch_axis = range(1, epochs + 1)
    plt.plot(epoch_axis, history_global_loss, label=f'{model_name} Train Global Loss', color='black', linewidth=3)
    if any(v is not None for v in history_val_global_loss):
        val_x = [i + 1 for i, v in enumerate(history_val_global_loss) if v is not None]
        val_y = [v for v in history_val_global_loss if v is not None]
        plt.plot(val_x, val_y, label=f'{model_name} Val Global Loss', color='red', linewidth=2)

    for g_key, losses in history_edge_loss.items():
        valid_epochs = [i + 1 for i, l in enumerate(losses) if l is not None]
        valid_losses = [l for l in losses if l is not None]

        if valid_losses:
            edge_name = f"{g_key[0]} <-> {g_key[2]}"
            plt.plot(valid_epochs, valid_losses, label=f"Raw Loss: {edge_name}", linestyle='--', marker='.', alpha=0.8)

    plt.title(f'{model_name} GNN Pretraining Loss vs Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('BCE Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    os.makedirs('./results', exist_ok=True)
    save_path = f'./results/pretrain_{model_name.lower()}_loss_curve.png'
    plt.savefig(save_path, dpi=300)
    print(f"🖼️ {model_name} 折线图已成功保存至: {save_path}")

    return gnn_model


def train_gnn():
    print("📂 正在加载图数据集...")
    try:
        dataset = torch.load('offline_graphs_dataset.pt', weights_only=False)
    except FileNotFoundError:
        print("❌ 未找到 offline_graphs_dataset.pt，请先运行 collect_data.py！")
        return

    mcs_dataset = dataset['mcs']
    iev_dataset = dataset['iev']
    _validate_dataset_feature_dims(mcs_dataset, iev_dataset)

    print("\n" + "=" * 60)
    print("🎯 开始训练 MCS-GNN")
    print("=" * 60)
    train_single_gnn(mcs_dataset, MCSHeteroGNN, "MCS", hidden_channels=HIDDEN_DIM)

    print("\n" + "=" * 60)
    print("🎯 开始训练 IEV-GNN")
    print("=" * 60)
    train_single_gnn(iev_dataset, IEVHeteroGNN, "IEV", hidden_channels=HIDDEN_DIM)


if __name__ == '__main__':
    train_gnn()
