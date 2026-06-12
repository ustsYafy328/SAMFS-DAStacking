from dataclasses import dataclass


TARGET_COL = "挤压速度"

ALL_FEATURES = [
    "生产批产量",
    "加工长度",
    "短棒长度",
    "产品米重",
    "生产批产量_全局标准化偏差",
    "短棒长度_全局标准化偏差",
    "短棒长度_低",
    "型材表面处理方式_未知",
    "短棒长度_高",
    "模具类型_1.0",
    "铝棒上机温度",
    "型材表面处理方式_喷涂",
    "型材表面处理方式_光身",
    "铝棒上机温度_全局标准化偏差",
    "模具类型_16.0",
    "生产批产量_高",
    "合金牌号",
    "型材表面处理方式_氧化",
    "模具类型_6.0",
    "短棒长度_中",
    "生产批产量_中",
    "铝棒上机温度_低",
    "铝棒上机温度_高",
    "模具类型_2.0",
    "一出几",
    "型材表面处理方式_氟碳",
    "生产批产量_低",
    "铝棒上机温度_中",
    "模具上机温度",
    "模具直径",
]

# The optimal M=17 subset reported in the paper.
PAPER_SAMFS_FEATURES = [
    "生产批产量",
    "加工长度",
    "短棒长度",
    "产品米重",
    "生产批产量_全局标准化偏差",
    "短棒长度_全局标准化偏差",
    "短棒长度_低",
    "型材表面处理方式_未知",
    "短棒长度_高",
    "模具类型_1.0",
    "铝棒上机温度",
    "型材表面处理方式_喷涂",
    "型材表面处理方式_光身",
    "铝棒上机温度_全局标准化偏差",
    "模具类型_16.0",
    "生产批产量_高",
    "合金牌号",
]


@dataclass
class DAStackingConfig:
    random_state: int = 43
    selected_feature_count: int = 17
    actor_hidden: tuple = (256, 160, 96)
    critic_hidden: tuple = (512, 320, 192)
    batch_size: int = 128
    max_steps: int = 40000
    gamma: float = 0.9
    tau: float = 1e-3
    actor_lr: float = 1e-4
    critic_lr: float = 5e-4
    buffer_size: int = 20000
    start_steps: int = 2048
    update_every: int = 5
    updates_per_step: int = 12
    eval_every: int = 200
    patience_evals: int = 25
    ent_coeff: float = 0.0045
    reward_lambda: float = 0.8
    dropout: float = 0.28

