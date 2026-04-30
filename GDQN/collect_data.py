import random

import torch

from env.config import *
from env.core import MCS
from env.environment import MultiAgentEnv
from env.utils import haversine
from env.world import World
from GDQN.net import MCSHeteroGNN, IEVHeteroGNN


def collect_offline_graphs(conf, num_episodes=100, max_steps=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_mcs_gnn = MCSHeteroGNN(hidden_channels=32).to(device)
    dummy_iev_gnn = IEVHeteroGNN(hidden_channels=32).to(device)
    dummy_mcs_gnn.eval()
    dummy_iev_gnn.eval()
    dummy_gnn = {'mcs': dummy_mcs_gnn, 'iev': dummy_iev_gnn}
    world = World(conf, dummy_gnn, device)
    env = MultiAgentEnv(world)

    mcs_graph_dataset = []
    iev_graph_dataset = []

    print("开始收集图数据...")
    for ep in range(num_episodes):
        env.reset()
        for step in range(max_steps):
            action_n = []
            for agent in env.world.agents:
                if isinstance(agent, MCS):
                    strategy = random.random()

                    if strategy < 0.33:
                        action_n.append(agent.pos)
                    elif strategy < 0.66:                # 从附近quasi中随机选取一个
                        nearby_quasis = []
                        for ev in env.world.quasi_iev:
                            dist = haversine(agent.pos[0], agent.pos[1], ev.pos[0], ev.pos[1])
                            if dist < COMM_REGION_R:
                                nearby_quasis.append(ev)
                        if len(nearby_quasis) > 0:
                            random_target = random.choice(nearby_quasis)
                            action_n.append(random_target.pos)
                        else:
                            action_n.append(agent.pos)
                    else:                                # 选取最近的quasi
                        min_dist = 999
                        target_pos = agent.pos
                        for ev in env.world.quasi_iev:
                            dist = haversine(agent.pos[0], agent.pos[1], ev.pos[0], ev.pos[1])
                            if dist < min_dist:
                                min_dist = dist
                                target_pos = ev.pos
                        action_n.append(target_pos)
                else:                                    # 沿着轨迹形式
                    if agent.track_index + 1 < len(agent.track):
                        agent.track_index += 1
                        action_n.append(agent.track[agent.track_index])
                    else:
                        action_n.append(agent.pos)

            # 与训练一致：先完成环境一步交互，再从新一轮agents状态中取图
            # step: update -> step_finish -> match_and_get_neibor -> get_obs_n/reward
            _new_obs_n, _old_obs_n, _reward_n = env.step(action_n)
            graph_data_dict = env.world.get_global_state()

            if ep == 0 and step % 10 == 0:
                print(f"\n[{'=' * 15} Episode {ep} | Step {step} MCS图结构统计 {'=' * 15}]")
                mcs_hetero = graph_data_dict['mcs'].hetero_data
                total_nodes = 0
                print("🔵 节点分布 (Nodes):")
                for n_type in mcs_hetero.node_types:
                    n_count = mcs_hetero[n_type].num_nodes
                    total_nodes += n_count
                    print(f"  ├─ {n_type:<6} : {n_count} 个")
                print(f"  └─ 节点总数 : {total_nodes} 个")

                print("\n🔗 边分布 (Edges):")
                total_edges = 0
                for e_type in mcs_hetero.edge_types:
                    e_count = mcs_hetero[e_type].num_edges
                    total_edges += e_count
                    edge_str = f"{e_type[0]} -> {e_type[1]} -> {e_type[2]}"
                    print(f"  ├─ {edge_str:<25} : {e_count} 条")
                print(f"  └─ 边总数   : {total_edges} 条")

                print(f"\n[{'=' * 15} Episode {ep} | Step {step} IEV图结构统计 {'=' * 15}]")
                iev_hetero = graph_data_dict['iev'].hetero_data
                total_nodes = 0
                print("🔵 节点分布 (Nodes):")
                for n_type in iev_hetero.node_types:
                    n_count = iev_hetero[n_type].num_nodes
                    total_nodes += n_count
                    print(f"  ├─ {n_type:<6} : {n_count} 个")
                print(f"  └─ 节点总数 : {total_nodes} 个")

                print("\n🔗 边分布 (Edges):")
                total_edges = 0
                for e_type in iev_hetero.edge_types:
                    e_count = iev_hetero[e_type].num_edges
                    total_edges += e_count
                    edge_str = f"{e_type[0]} -> {e_type[1]} -> {e_type[2]}"
                    print(f"  ├─ {edge_str:<25} : {e_count} 条")
                print(f"  └─ 边总数   : {total_edges} 条")
                print("=" * 60 + "\n")

            mcs_graph_dataset.append(graph_data_dict['mcs'].hetero_data.cpu())
            iev_graph_dataset.append(graph_data_dict['iev'].hetero_data.cpu())

        if (ep + 1) % 10 == 0:
            print(f"已收集 {ep + 1} 个 Episodes, MCS图: {len(mcs_graph_dataset)} 张, IEV图: {len(iev_graph_dataset)} 张")

    torch.save({'mcs': mcs_graph_dataset, 'iev': iev_graph_dataset}, 'offline_graphs_dataset.pt')
    print("图数据收集完成并保存！")


if __name__ == '__main__':
    collect_offline_graphs(conf)
