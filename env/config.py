import os

# ==================== 环境常量 ====================
COMM_REGION_R = 2.0        # 通信范围半径，单位 km
LOWEST_POWER = 1.0         # 最低电量阈值，单位 kWh
MCS_SPEED = 11.0           # MCS移动速度，单位 m/s
NUM_TRAFFIC_NODES = 338    # 交通节点数量

# 价格常量
CHARGE_PRICE = 1.6         # 充电价格，单位 元/kWh
MCS_CHARGE_PRICE = 0.5     # MCS充电价格，单位 元/kWh
POWER_UNIT = 0.3           # 单位距离耗电，单位 kWh/km

# 图节点特征维度
VEHICLE_FEAT_DIM = 6
MCS_FEAT_DIM = 7
HIDDEN_DIM = 32

# 数据路径
PATH_INIT_EV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "track", "2014080"))  # 初始EV数据路径前缀

# ==================== 配置字典 ====================
conf = {
    # -----------------------------
    # 运行模式
    # -----------------------------
    "MCS_RANDOM": False,
    'IEV_RANDOM': False,
    
    # -----------------------------
    # 环境参数（World / Env）
    # -----------------------------
    "LOW_POWER": 8,                  # 低电量阈值：当EV剩余电量低于此值时，触发充电请求
    "MAX_WAIT_TIME": 2,              # 最大等待时间：EV发出充电请求后的最大容忍等待步数step
    "CHARGE_SPEED": 120,             # 充电速度：MCS为EV充电的速度，单位 kWh/hour
    "NUM_MCS": 20,                   # MCS数量：移动充电站（Mobile Charging Station）的数量
    "NUM_EV": 500,                   # EV数量：电动汽车（Electric Vehicle）的数量
    "MEAN": 40,                      # EV初始电量均值：EV初始剩余电量的平均值，单位 kWh
    "STAND": 14,                     # EV初始电量标准差：EV初始剩余电量的标准差，单位 kWh

    # -----------------------------
    # 训练参数（GNN-DQN）
    # -----------------------------
    "NUM_EPISODES": 101,              # 训练轮数：总共训练的episode数量
    "MAX_STEPS_PER_EPISODE": 100,    # 每轮最大步数：每个episode的最大时间步数
    "SEED": 42,                      # 随机种子：用于保证实验可重复性
    "EPS_START": 0.2,
    "EPS_END": 0.005,
    "EPS_DECAY_EPISODES": 60,
    
    # 模型架构参数
    "HIDDEN_DIM": HIDDEN_DIM,                # 隐藏层维度：GNN隐藏层的特征维度
    "VEHICLE_FEAT_DIM": VEHICLE_FEAT_DIM,
    "MCS_FEAT_DIM": MCS_FEAT_DIM,
    "NUM_LAYERS": 2,                 # GNN层数：图神经网络的层数
    "NUM_HEADS": 4,                  # 注意力头数：GAT（图注意力网络）的注意力头数量
    "DROPOUT": 0.0,                  # Dropout率：防止过拟合的dropout概率
    
    # 优化器参数
    "LR": 0.005,                     # 学习率：Adam优化器的学习率
    "GRAD_CLIP_NORM": 5.0,           # 梯度裁剪范数：防止梯度爆炸的梯度裁剪阈值
    
    # DQN参数
    "GAMMA": 0.97,                    # 折扣因子：未来奖励的折现因子，用于计算TD目标
    "BUFFER_CAPACITY": 50000,        # 回放缓冲区容量：经验回放缓冲区的最大容量
    "BATCH_SIZE": 128,               # 批次大小：每次从回放缓冲区采样的经验数量
    "WARMUP_STEPS": 500,             # 预热步数：开始训练前需要收集的经验步数
    "TARGET_UPDATE_INTERVAL": 50,    # 目标网络更新间隔：每隔多少步更新一次目标网络
    
    # 奖励函数参数
    "W_CHARGE": 3.0,                 # 充电权重：奖励函数中充电收益的权重
    "W_MOVE": 1.0,                   # 移动权重：奖励函数中移动成本的权重
    "REWARD_SCALE": 0.01,            # 奖励缩放因子：对最终奖励进行缩放的因子

    # -----------------------------
    # 推理/测试参数
    # -----------------------------
    # Top-K 候选过滤：MCS只考虑最近的K个quasi EV，IEV只考虑最近的K个task MCS
    "TOP_K_MCS_CANDIDATES": 5,
    "TOP_K_IEV_CANDIDATES": 5,

    "TEST_SEED": 2026,
    "TEST_MAX_STEPS_PER_EPISODE": 100,  # 测试每轮最大步数：测试时每个episode的最大时间步数
    "TEST_CKPT_PATH": os.path.join(os.path.dirname(__file__), "models", "gnn_dqn_best.pt"),  # 测试模型路径
}
