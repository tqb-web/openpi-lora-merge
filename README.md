# OpenPI LoRA Merge

一个用于合并 [OpenPI](https://github.com/Physical-Intelligence/openpi) 具身智能（VLA）模型 LoRA 权重并导出为 JAX/Flax 格式独立检查点（Checkpoints）的实用工具。

## 📌 功能特点

* **LoRA 权重熔断**：自动将训练过程中产生的 LoRA 旁路权重合并回 PaliGemma 主干网络。
* **原生 JAX 格式保存**：转换过程中保持 JAX/Flax 原生的矩阵维度（不进行 PyTorch 维度的 `.T` 转置），方便后续在 JAX 生态中直接进行微调或推理。
---

## 🛠️ 环境准备

在运行此脚本之前，请确保你已经激活了包含 OpenPI 依赖的虚拟环境：

```bash
# 激活你的虚拟环境
source /path/to/your/openpi/.venv/bin/shift/activate

```

---

## 🚀 快速上手

### 1. 运行合并脚本

你可以直接使用项目提供的 Bash 脚本来运行合并操作。首先赋予脚本执行权限：

```bash
chmod +x run_merge.sh

```

然后执行脚本：

```bash
./run_merge.sh

```

### 2. 脚本参数说明

如果你想直接通过命令行传递参数，可以参考以下格式：

```bash
python3 scripts/merge_lora1.py \
    --repo-root "" \
    --checkpoint_dir "" \
    --config_name "" \
    --output_path "" \
    --precision ""

```

**参数详解：**

| 参数名 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--repo-root` | `str` | 必填 | OpenPI 源码仓库的根目录路径 |
| `--checkpoint_dir` | `str` | 必填 | 训练产生的含有 LoRA 权重的原始 Checkpoint 目录 |
| `--config_name` | `str` | 必填 | 模型的配置文件名称（例如 `pi05_libero`） |
| `--output_path` | `str` | `output` | 合并后新 JAX 权重的保存输出目录 |
| `--precision` | `str` | `float32` | 导出权重的精度，可选 `float32`, `bfloat16` |

---

## 🤝 贡献与反馈

欢迎提交 Issue 或 Pull Request 来优化这个合并工具！