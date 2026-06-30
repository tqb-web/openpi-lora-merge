#!/bin/bash

REPO_ROOT="" # openpi仓库根目录路径
CHECKPOINT_DIR="" # 模型权重文件夹路径
CONFIG_NAME="" # 配置文件名
OUTPUT_PATH="" # 合并后的模型输出路径
PRECISION="" # bfloat16 or float32

# 执行合并脚本
python3 scripts/merge_lora1.py \
    --repo-root "$REPO_ROOT" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --config_name "$CONFIG_NAME" \
    --output_path "$OUTPUT_PATH" \
    --precision "$PRECISION"