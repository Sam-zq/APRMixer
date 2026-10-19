# APRMixer 代码整理包

包含核心模型、经验参考与数据预处理、完整训练入口、验证选模、测试评估、批量实验脚本和合成数据检查。英文详细说明见 [README.md](README.md)。

## 快速运行

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/inference.py
python train.py --data-root ./datasets --station 1 --horizon 96 --seed 2021 --device cuda:0
python evaluate.py --data-root ./datasets --station 1 --horizon 96 --seed 2021 --device cuda:0
```

实际数据需自行放入 `datasets/`，格式见 [数据说明](datasets/README.md)。只读示例和检查使用合成数据，不代表真实预测效果。

## 目录

| 路径 | 内容 |
|---|---|
| `models/APRMixer.py` | 只包含当前有效结构的精简模型，用于阅读和推理 |
| `reproduction/APRMixer.py` | 保留原实验初始化顺序的模型，默认用于训练 |
| `data_provider/` | 训练集拟合的经验参考、归一化与时间窗口构建 |
| `training/engine.py` | 普通全天 MSE 训练、验证选模与评估流程 |
| `evaluation/` | 全天、白天和爬坡指标 |
| `configuration.py` | 最终模型参数、站点容量和训练设置 |
| `train.py` / `evaluate.py` | 单次训练及保存权重后的重新评估 |
| `run_experiments.py` | 顺序运行站点、预测长度和随机种子的组合 |
| `tests/` / `examples/` | 合成数据检查及接口示例 |
| `docs/` | 复现边界、源码来源、测试记录与上传指南 |

默认预测长度 16/32/48/96 对应 4/8/12/24 小时。默认全天 MSE 训练，按全天验证 MSE 选择权重。两个模型版本能互相严格加载权重并得到相同推理输出，但精简版删除了历史构造过程，因此相同随机种子下的重新训练初始化不相同。

本包只包含 APRMixer 自身的完整训练和评估链路，不包含外部对比模型、所有消融实验或论文完整结果表。真实数据、权重和论文稿未打包。

