#!/bin/bash

# 检查参数数量是否正确
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

# 获取输入和输出目录
INPUT_DIR=$1
OUTPUT_DIR=$2

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 运行Python脚本，传递输入和输出目录
python3 base_split.py "$INPUT_DIR" "$OUTPUT_DIR"

# 检查Python脚本是否成功运行
if [ $? -eq 0 ]; then
    echo "Processing completed successfully."
else
    echo "Error: Python script failed."
    exit 1
fi
