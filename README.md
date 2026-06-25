# 地址级 AI 学习 — Address-Level AI Learning

> **T×P×I 三维地址行为建模 + 群体智能推演交易系统**

## 🎯 核心构想

链上交易的本质是多重映射：一个人 ↔ 多个地址，多个地址 ↔ 多个人。  
本项目的目标是将每个链上地址的交易历史作为「token sequence」来建模，类比 LLM 的训练范式。

### 三层架构

| 层级 | 功能 | 技术组件 |
|------|------|---------|
| **Layer 3** | 群体智能推演 | MiroFish 数字平行世界 + Forum 论坛机制 |
| **Layer 2** | 地址 LoRA 集群 | T×P×I 训练 → 地址专属 LoRA → 策略族群聚类 |
| **Layer 1** | 链上数据层 | 地址实体解析 + T×P×I 三维张量 |

### 三个核心维度（T×P×I）

- **时间 (Time)** — 区块时间戳、交易间隔、市场阶段
- **行为 (Action)** — 交易方向（多/空/平）、仓位大小、杠杆
- **价格 (Price)** — 入场价/出场价、相对价格变化率

### 技术路线

```mermaid
flowchart LR
    A[链上地址] --> B[T×P×I 序列化]
    B --> C[Base Model 预训练]
    C --> D[每地址 LoRA 微调]
    D --> E[策略族群聚类]
    E --> F[MiroFish 群体推演]
    F --> G[交易信号]
```

## 🚀 快速开始

```bash
# 安装依赖
uv sync

# 1. 获取 Hyperliquid 地址数据
python scripts/fetch_data.py --top-n 100 --days 90

# 2. 训练基线模型
python scripts/train.py --model lstm --epochs 50

# 3. 评估
python scripts/evaluate.py --model checkpoints/lstm_best.pt
```

## 📁 项目结构

```
address-lora-trading/
├── pyproject.toml          # 项目配置与依赖
├── data/                   # 数据目录
│   ├── raw/                # 原始 Hyperliquid API 数据
│   └── processed/          # 预处理后的 T×P×I 序列
├── src/
│   ├── data/
│   │   ├── hyperliquid.py  # Hyperliquid API 数据抓取
│   │   └── preprocess.py   # T×P×I 序列化预处理
│   ├── models/
│   │   ├── baseline.py     # Markov Chain 基线
│   │   └── lstm_model.py   # LSTM 序列模型
│   └── eval/
│       └── metrics.py      # 评估指标
├── scripts/
│   ├── fetch_data.py       # 数据获取入口
│   ├── train.py            # 模型训练入口
│   └── evaluate.py         # 模型评估入口
└── notebooks/              # Jupyter 实验笔记
```

## 📊 评估指标

- **Direction Accuracy**: 预测交易方向（多/空）的准确率
- **Size MAE**: 仓位大小的平均绝对误差
- **Sequence Perplexity**: 行为序列的困惑度
- **Baseline vs Model**: 与 Markov Chain 基线的对比

## 🧠 灵感来源

- [ArkStream Capital 关于地址级行为因子策略的讨论](https://x.com/i/status/2069704962368757788)
- [MiroFish - 群智智能预测引擎](https://github.com/666ghj/MiroFish)
- [BettaFish (微舆) - 多 Agent 舆情分析系统](https://github.com/666ghj/BettaFish)
- Decision Transformer (Chen et al. 2021)
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al. 2021)
