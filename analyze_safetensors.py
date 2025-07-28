#!/usr/bin/env python3
"""
SafeTensors文件分析脚本
用于读取和分析safetensors文件的元数据和结构
"""

import os
import sys
import hashlib
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

def get_file_md5(filepath):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def analyze_safetensors_file(filepath):
    """分析safetensors文件的详细信息"""
    print(f"分析文件: {filepath}")
    print(f"文件大小: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
    print(f"当前MD5: {get_file_md5(filepath)}")
    print("-" * 50)
    
    # 读取元数据
    with safe_open(filepath, framework="pt", device="cpu") as f:
        print("元数据信息:")
        metadata = f.metadata()
        if metadata:
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        else:
            print("  无元数据")
        print()
        
        print("张量信息:")
        tensor_names = list(f.keys())
        tensor_count = len(tensor_names)
        print(f"总张量数量: {tensor_count}")

        # 只显示最后一个张量的详细信息
        if tensor_count > 0:
            last_tensor_name = tensor_names[-1]
            last_tensor = f.get_tensor(last_tensor_name)

            print(f"\n最后一个张量详细信息:")
            print(f"  名称: {last_tensor_name}")
            print(f"  形状: {list(last_tensor.shape)}")
            print(f"  类型: {last_tensor.dtype}")
            print(f"  元素数: {last_tensor.numel()}")
            print(f"  大小: {last_tensor.numel() * last_tensor.element_size() / 1024:.2f} KB")

            # 显示统计信息
            if last_tensor.numel() > 0:
                print(f"  范围: [{last_tensor.min().item():.6f}, {last_tensor.max().item():.6f}]")
                print(f"  均值: {last_tensor.mean().item():.6f}")
                print(f"  标准差: {last_tensor.std().item():.6f}")

                # 显示最后几个元素的值
                flat_tensor = last_tensor.flatten()
                last_elements = flat_tensor[-5:] if flat_tensor.numel() >= 5 else flat_tensor
                print(f"  最后5个元素: {[f'{x.item():.6f}' for x in last_elements]}")

        return tensor_names, last_tensor_name if tensor_count > 0 else None

# 删除不需要的函数

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python analyze_safetensors.py <safetensors文件路径>")
        print("示例: python analyze_safetensors.py realcomic_000000900.safetensors")
        exit(1)

    filepath = sys.argv[1]

    if not os.path.exists(filepath):
        print(f"错误: 文件 {filepath} 不存在")
        exit(1)

    try:
        # 分析文件
        tensor_names, last_tensor_name = analyze_safetensors_file(filepath)

        print("\n" + "="*50)
        print("修改策略建议:")
        print("1. 选择最后一个张量进行微小修改")
        print("2. 修改幅度建议: 在最后一个元素上加/减一个极小值(如1e-8)")
        print("3. 这样可以确保MD5改变，但对模型性能影响最小")
        if last_tensor_name:
            print(f"4. 推荐修改目标: {last_tensor_name}")
        print("="*50)

    except Exception as e:
        print(f"错误: {e}")
        print("请确保已安装safetensors库: pip install safetensors")
