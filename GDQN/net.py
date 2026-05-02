import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_max
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GATConv, HeteroConv

from env.config import VEHICLE_FEAT_DIM, MCS_FEAT_DIM


class MCSHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels=32, dropout=0.0, heads=2, num_layers=2):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.dropout = float(dropout)
        self.heads = int(heads)
        self.num_layers = int(num_layers)

        self.lin_dict = torch.nn.ModuleDict({
            "idle": Linear(MCS_FEAT_DIM, hidden_channels),
            "quasi": Linear(VEHICLE_FEAT_DIM, hidden_channels),
            "task": Linear(MCS_FEAT_DIM, hidden_channels),
        })

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                ("idle", "info", "idle"): GATConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // heads,
                    heads=heads,
                    add_self_loops=False,
                ),
                ("idle", "info", "quasi"): GATConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // heads,
                    heads=heads,
                    add_self_loops=False,
                ),
                ("quasi", "info", "idle"): GATConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // heads,
                    heads=heads,
                    add_self_loops=False,
                ),
                ("quasi", "info", "task"): GATConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // heads,
                    heads=heads,
                    add_self_loops=False,
                ),
                ("task", "info", "quasi"): GATConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // heads,
                    heads=heads,
                    add_self_loops=False,
                ),
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.norms = torch.nn.ModuleList([
            torch.nn.ModuleDict({k: torch.nn.LayerNorm(hidden_channels) for k in self.lin_dict.keys()})
            for _ in range(num_layers)
        ])

    def forward(self, hetero_data):
        device = next(self.parameters()).device

        x_dict = hetero_data.x_dict
        edge_index_dict = hetero_data.edge_index_dict

        x_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x_dict.items()}
        edge_index_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in edge_index_dict.items()}

        h_dict = {}
        for node_type, lin in self.lin_dict.items():
            x = x_dict.get(node_type)
            if x is not None and x.size(0) > 0:
                h_dict[node_type] = lin(x).relu()
            else:
                h_dict[node_type] = torch.zeros((0, self.hidden_channels), device=device)

        for i, conv in enumerate(self.convs):
            model_etypes = conv.convs.keys()
            active_edges = {
                etype: eidx for etype, eidx in edge_index_dict.items()
                if etype in model_etypes and eidx is not None and eidx.numel() > 0
            }

            h_dict_new = conv(h_dict, active_edges)

            combined = {}
            for node_type in self.lin_dict.keys():
                if node_type in h_dict_new:
                    x = h_dict_new[node_type] + h_dict[node_type]
                else:
                    x = h_dict[node_type]

                if x.size(0) > 0:
                    x = self.norms[i][node_type](x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

                combined[node_type] = x

            h_dict = combined

        return h_dict


class IEVHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels=32, dropout=0.0, heads=2, num_layers=2):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.dropout = float(dropout)
        self.heads = int(heads)
        self.num_layers = int(num_layers)

        self.lin_dict = torch.nn.ModuleDict({
            "iev": Linear(VEHICLE_FEAT_DIM, hidden_channels),
            "task": Linear(MCS_FEAT_DIM, hidden_channels),
        })

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                ("iev", "info", "iev"): GATConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // heads,
                    heads=heads,
                    add_self_loops=False,
                ),
                ("iev", "info", "task"): GATConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // heads,
                    heads=heads,
                    add_self_loops=False,
                ),
                ("task", "info", "iev"): GATConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // heads,
                    heads=heads,
                    add_self_loops=False,
                ),
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.norms = torch.nn.ModuleList([
            torch.nn.ModuleDict({k: torch.nn.LayerNorm(hidden_channels) for k in self.lin_dict.keys()})
            for _ in range(num_layers)
        ])

    def forward(self, hetero_data):
        device = next(self.parameters()).device

        x_dict = hetero_data.x_dict
        edge_index_dict = hetero_data.edge_index_dict

        x_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x_dict.items()}
        edge_index_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in edge_index_dict.items()}

        h_dict = {}
        for node_type, lin in self.lin_dict.items():
            x = x_dict.get(node_type)
            if x is not None and x.size(0) > 0:
                h_dict[node_type] = lin(x).relu()
            else:
                h_dict[node_type] = torch.zeros((0, self.hidden_channels), device=device)

        for i, conv in enumerate(self.convs):
            model_etypes = conv.convs.keys()
            active_edges = {
                etype: eidx for etype, eidx in edge_index_dict.items()
                if etype in model_etypes and eidx is not None and eidx.numel() > 0
            }

            h_dict_new = conv(h_dict, active_edges)

            combined = {}
            for node_type in self.lin_dict.keys():
                if node_type in h_dict_new:
                    x = h_dict_new[node_type] + h_dict[node_type]
                else:
                    x = h_dict[node_type]

                if x.size(0) > 0:
                    x = self.norms[i][node_type](x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

                combined[node_type] = x

            h_dict = combined

        return h_dict


class DynamicQNetwork(nn.Module):
    """Q值解码器：直接消费 GNN 聚合特征 + 原始特征，输出每个候选动作的 Q 值。

    输入 (obs_feat):
        self_h    [H] 或 [B, H]  — 源节点 GNN 聚合嵌入
        target_h  [N, H] 或 [B, H]  — 目标节点 GNN 聚合嵌入
        self_raw  [S] 或 [B, S]  — 源节点原始特征 (GNN 编码前)
        target_raw [N, T] 或 [B, T] — 目标节点原始特征

    内部拼接 5×H 特征后由 MLP 打出标量 Q 值。
    """

    def __init__(self, hidden_dim=32, src_raw_dim=7, dst_raw_dim=6):
        super(DynamicQNetwork, self).__init__()

        self.src_raw_proj = nn.Linear(src_raw_dim, hidden_dim)
        self.dst_raw_proj = nn.Linear(dst_raw_dim, hidden_dim)

        # self_h[H] + target_h[H] + src_proj[H] + dst_proj[H] + interaction[H]
        final_input_dim = hidden_dim * 5

        self.norm = nn.LayerNorm(final_input_dim)

        self.decoder = nn.Sequential(
            nn.Linear(final_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs_feat):
        device = next(self.parameters()).device
        self_h = obs_feat['self_h'].to(device)
        target_h = obs_feat['target_h'].to(device)
        self_raw = obs_feat['self_raw'].to(device)
        target_raw = obs_feat['target_raw'].to(device)

        N = target_h.shape[0]
        if N == 0:
            return torch.empty(0, device=device)

        # 线性投影原始特征到 hidden_dim
        self_raw_proj = self.src_raw_proj(self_raw)
        target_raw_proj = self.dst_raw_proj(target_raw)

        # 单样本 → 打包模式对齐: self 特征扩展到 N 份
        if self_h.dim() == 1:
            self_h = self_h.unsqueeze(0).expand(N, -1)
        if self_raw_proj.dim() == 1:
            self_raw_proj = self_raw_proj.unsqueeze(0).expand(N, -1)

        # Hadamard 积: 显式建模源-目标交互
        interaction = self_raw_proj * target_raw_proj

        combined = torch.cat([
            self_h,
            target_h,
            self_raw_proj,
            target_raw_proj,
            interaction,
        ], dim=-1)

        normed = self.norm(combined)
        q_values = self.decoder(normed).squeeze(-1)
        return q_values


class GDQNNet(object):
    def __init__(self, conf, gnn=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn = gnn

        # 超参数
        self.lr = conf.get('LR', 0.005)
        self.gamma = conf.get('GAMMA', 0.9)
        self.target_replace_iter = conf.get('TARGET_UPDATE_INTERVAL', 100)
        self.tau = 0.005  # 软更新系数
        self.grad_clip_norm = conf.get('GRAD_CLIP_NORM', 5.0)
        self.buffer_capacity = conf.get('BUFFER_CAPACITY', 50000)
        self.batch_size = conf.get('BATCH_SIZE', 128)
        self.warmup_steps = conf.get('WARMUP_STEPS', 500)
        self.learn_every = conf.get('LEARN_EVERY', 1)

        self.hidden_dim = conf.get('HIDDEN_DIM', 32)

        # ----------------- MCS 网络 (src=idle_MCS[7维], dst=quasi_EV[6维]) -----------------
        self.mcs_eval_net = DynamicQNetwork(
            self.hidden_dim, src_raw_dim=MCS_FEAT_DIM, dst_raw_dim=VEHICLE_FEAT_DIM).to(self.device)
        self.mcs_target_net = DynamicQNetwork(
            self.hidden_dim, src_raw_dim=MCS_FEAT_DIM, dst_raw_dim=VEHICLE_FEAT_DIM).to(self.device)
        self.mcs_optimizer = torch.optim.Adam(self.mcs_eval_net.parameters(), lr=self.lr)
        self.mcs_learn_step_counter = 0

        # ----------------- IEV 网络 (src=IEV[6维], dst=task_MCS[7维]) -----------------
        self.iev_eval_net = DynamicQNetwork(
            self.hidden_dim, src_raw_dim=VEHICLE_FEAT_DIM, dst_raw_dim=MCS_FEAT_DIM).to(self.device)
        self.iev_target_net = DynamicQNetwork(
            self.hidden_dim, src_raw_dim=VEHICLE_FEAT_DIM, dst_raw_dim=MCS_FEAT_DIM).to(self.device)
        self.iev_optimizer = torch.optim.Adam(self.iev_eval_net.parameters(), lr=self.lr)
        self.iev_learn_step_counter = 0

        self.loss_func = nn.MSELoss()
        self.mcs_replay = deque(maxlen=self.buffer_capacity)
        self.iev_replay = deque(maxlen=self.buffer_capacity)
        self.total_env_steps = 0

        # 初始化同步 Target 网络
        self.mcs_target_net.load_state_dict(self.mcs_eval_net.state_dict())
        self.iev_target_net.load_state_dict(self.iev_eval_net.state_dict())

    def mcs_choose_action(self, agent_obs, epsilon):
        """MCS 智能体动作选择（经典 epsilon-greedy：epsilon 为随机探索概率）"""
        feat = agent_obs['feat']
        if feat['target_h'].shape[0] == 0:
            return None  # 此时无侯选动作

        if np.random.uniform() < epsilon:
            # 以 epsilon 概率随机探索
            action_idx = np.random.randint(0, feat['target_h'].shape[0])
        else:
            # 以 1-epsilon 概率利用当前网络
            with torch.no_grad():
                q_values = self.mcs_eval_net(feat)
                action_idx = torch.argmax(q_values).item()

        return action_idx

    def iev_choose_action(self, agent_obs, epsilon):
        """IEV 智能体动作选择（经典 epsilon-greedy：epsilon 为随机探索概率）"""
        feat = agent_obs['feat']
        if feat['target_h'].shape[0] == 0:
            return None

        if np.random.uniform() < epsilon:
            action_idx = np.random.randint(0, feat['target_h'].shape[0])
        else:
            with torch.no_grad():
                q_values = self.iev_eval_net(feat)
                action_idx = torch.argmax(q_values).item()

        return action_idx

    def _learn_base(self, batch, eval_net, target_net, optimizer, step_counter):
        states, actions, rewards, next_states, dones = batch

        # 筛掉 next_state 非终止但无可选动作的样本 (无法做 TD bootstrap, 保留会低估 Q)
        keep_idx = []
        for i in range(len(states)):
            n_feat = next_states[i]['feat']
            if dones[i] or n_feat['target_h'].shape[0] > 0:
                keep_idx.append(i)
        if len(keep_idx) < len(states):
            states = [states[i] for i in keep_idx]
            actions = [actions[i] for i in keep_idx]
            rewards = [rewards[i] for i in keep_idx]
            next_states = [next_states[i] for i in keep_idx]
            dones = [dones[i] for i in keep_idx]
        if len(states) == 0:
            return 0.0

        # 1. 目标网络软更新 (Polyak averaging)
        if step_counter % self.target_replace_iter == 0:
            for target_param, eval_param in zip(target_net.parameters(), eval_net.parameters()):
                target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)

        # 2) 当前状态并行计算 Q(s,a)
        packed_state_feat, state_offsets = self._pack_feats(states)
        q_all_flat = eval_net(packed_state_feat)  # [sum(N_i)]
        action_indices = torch.tensor(actions, dtype=torch.long, device=self.device)
        action_offsets = torch.tensor(state_offsets, dtype=torch.long, device=self.device)
        q_evals = q_all_flat[action_offsets + action_indices]

        # 3) Double DQN 目标并行计算
        q_targets = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        valid_next_idx = []
        valid_next_states = []
        for i, n_state in enumerate(next_states):
            n_feat = n_state['feat']
            if (not dones[i]) and n_feat['target_h'].shape[0] > 0:
                valid_next_idx.append(i)
                valid_next_states.append(n_state)

        if valid_next_states:
            packed_next_feat, _ = self._pack_feats(valid_next_states)
            next_batch_idx = packed_next_feat['batch_idx']

            with torch.no_grad():
                q_next_eval_flat = eval_net(packed_next_feat)      # 用于选动作
                q_next_target_flat = target_net(packed_next_feat)  # 用于评估动作

                _, argmax_pos = scatter_max(
                    q_next_eval_flat,
                    next_batch_idx,
                    dim=0,
                    dim_size=len(valid_next_states),
                )
                chosen_q_next = q_next_target_flat[argmax_pos]
                valid_rewards = torch.tensor([rewards[i] for i in valid_next_idx],
                                             dtype=torch.float32, device=self.device)
                q_targets_valid = valid_rewards + self.gamma * chosen_q_next

            q_targets[torch.tensor(valid_next_idx, dtype=torch.long, device=self.device)] = q_targets_valid

        # 3. 梯度下降
        loss = self.loss_func(q_evals, q_targets)
        optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(eval_net.parameters(), self.grad_clip_norm)
        optimizer.step()

        return loss.item()

    def _pack_feats(self, obs_list):
        target_h_list = []
        self_h_list = []
        target_raw_list = []
        self_raw_list = []
        batch_idx_list = []
        offsets = []
        running = 0

        for i, obs in enumerate(obs_list):
            feat = obs['feat']
            n = feat['target_h'].shape[0]
            offsets.append(running)
            if n == 0:
                continue

            running += n
            target_h = feat['target_h'].to(self.device)
            self_h = feat['self_h'].to(self.device).unsqueeze(0).expand(n, -1)
            target_raw = feat['target_raw'].to(self.device)
            self_raw = feat['self_raw'].to(self.device).unsqueeze(0).expand(n, -1)

            target_h_list.append(target_h)
            self_h_list.append(self_h)
            target_raw_list.append(target_raw)
            self_raw_list.append(self_raw)
            batch_idx_list.append(torch.full((n,), i, dtype=torch.long, device=self.device))

        if not target_h_list:
            return {
                'self_h': torch.zeros((0, self.hidden_dim), device=self.device),
                'target_h': torch.zeros((0, self.hidden_dim), device=self.device),
                'self_raw': torch.empty((0, 0), device=self.device),
                'target_raw': torch.empty((0, 0), device=self.device),
                'batch_idx': torch.zeros((0,), dtype=torch.long, device=self.device),
            }, offsets

        packed_feat = {
            'self_h': torch.cat(self_h_list, dim=0),
            'target_h': torch.cat(target_h_list, dim=0),
            'self_raw': torch.cat(self_raw_list, dim=0),
            'target_raw': torch.cat(target_raw_list, dim=0),
            'batch_idx': torch.cat(batch_idx_list, dim=0),
        }
        return packed_feat, offsets

    def mcs_learn(self, mcs_batch):
        loss = self._learn_base(mcs_batch, self.mcs_eval_net, self.mcs_target_net,
                                self.mcs_optimizer, self.mcs_learn_step_counter)
        self.mcs_learn_step_counter += 1
        return loss

    def iev_learn(self, iev_batch):
        loss = self._learn_base(iev_batch, self.iev_eval_net, self.iev_target_net,
                                self.iev_optimizer, self.iev_learn_step_counter)
        self.iev_learn_step_counter += 1
        return loss

    def _sample_batch_from_buffer(self, buffer):
        samples = random.sample(buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return list(states), list(actions), list(rewards), list(next_states), list(dones)

    def push_transition(self, is_mcs, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if is_mcs:
            self.mcs_replay.append(transition)
        else:
            self.iev_replay.append(transition)

    def step_and_learn(self):
        self.total_env_steps += 1
        if self.total_env_steps < self.warmup_steps:
            return None, None
        if self.total_env_steps % self.learn_every != 0:
            return None, None

        mcs_loss = None
        iev_loss = None
        if len(self.mcs_replay) >= self.batch_size:
            mcs_batch = self._sample_batch_from_buffer(self.mcs_replay)
            mcs_loss = self.mcs_learn(mcs_batch)
        if len(self.iev_replay) >= self.batch_size:
            iev_batch = self._sample_batch_from_buffer(self.iev_replay)
            iev_loss = self.iev_learn(iev_batch)
        return mcs_loss, iev_loss

    def save_model(self, i_episode, conf):
        os.makedirs('./models', exist_ok=True)
        print(f"--> 保存模型: Episode {i_episode}")
        mcs_path = f"./models/mcs_net_ep{i_episode}.pkl"
        iev_path = f"./models/iev_net_ep{i_episode}.pkl"
        torch.save(self.mcs_eval_net.state_dict(), mcs_path)
        torch.save(self.iev_eval_net.state_dict(), iev_path)
