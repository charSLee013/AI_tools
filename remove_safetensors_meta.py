#!/usr/bin/env python3
"""
SafeTensors元数据移除脚本
移除safetensors文件中的metadata信息，只保留张量数据
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

def analyze_metadata(input_file):
    """
    分析safetensors文件的metadata信息

    Args:
        input_file: 输入文件路径

    Returns:
        metadata: 元数据字典
    """
    # 使用safe_open来访问metadata
    with safe_open(input_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        tensor_keys = f.keys()

        print(f"张量数量: {len(tensor_keys)}")

        if metadata:
            print("发现的元数据:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            print(f"元数据条目数: {len(metadata)}")
        else:
            print("未发现元数据")

        return metadata

def remove_metadata_from_safetensors(input_file, output_file):
    """
    移除safetensors文件中的metadata并保存新文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("-" * 60)

    # 计算原始文件MD5
    original_md5 = get_file_md5(input_file)

    # 分析原始文件的metadata
    print("=== 原始文件Metadata分析 ===")
    original_metadata = analyze_metadata(input_file)

    # 加载所有张量数据（不包含metadata）
    print(f"\n加载张量数据...")
    tensors = load_file(input_file)

    # 计算总参数量
    total_params = sum(tensor.numel() for tensor in tensors.values())
    print(f"成功加载 {len(tensors)} 个张量，总参数量: {total_params:,}")

    # 保存文件（不包含metadata）
    print(f"\n保存无metadata的文件...")
    save_file(tensors, output_file)

    # 验证新文件
    new_md5 = get_file_md5(output_file)
    print(f"\n=== 处理结果验证 ===")

    # 分析新文件的metadata
    print(f"新文件metadata检查:")
    new_metadata = analyze_metadata(output_file)
    
    # 文件大小对比
    original_size = os.path.getsize(input_file)
    new_size = os.path.getsize(output_file)
    size_diff = original_size - new_size

    print(f"\n=== Metadata处理结果 ===")
    print(f"原始metadata条目: {len(original_metadata) if original_metadata else 0}")
    print(f"新文件metadata条目: {len(new_metadata) if new_metadata else 0}")

    if original_metadata and not new_metadata:
        print("✅ 成功移除所有metadata")
    elif original_metadata and new_metadata:
        print("⚠️  部分metadata仍然存在")
    elif not original_metadata:
        print("ℹ️  原文件本身没有metadata")

    print(f"\n文件大小变化: {original_size / (1024*1024):.2f} MB → {new_size / (1024*1024):.2f} MB")
    if size_diff > 0:
        print(f"减少: {size_diff} 字节")
    print(f"MD5已改变: {'是' if original_md5 != new_md5 else '否'}")
    
    return original_metadata, new_metadata

def main():
    if len(sys.argv) < 2:
        print("用法: python remove_safetensors_meta.py <输入文件> [输出文件]")
        print("示例: python remove_safetensors_meta.py model.safetensors")
        print("示例: python remove_safetensors_meta.py input.safetensors output_clean.safetensors")
        exit(1)
    
    input_file = sys.argv[1]
    
    # 生成默认输出文件名
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_no_meta.safetensors"
    
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        exit(1)
    
    if os.path.exists(output_file):
        response = input(f"输出文件 {output_file} 已存在，是否覆盖? (y/N): ")
        if response.lower() != 'y':
            print("操作取消")
            exit(0)
    
    try:
        print("="*60)
        print("SafeTensors Metadata移除工具")
        print("="*60)
        
        original_metadata, _ = remove_metadata_from_safetensors(
            input_file, output_file
        )

        print("\n" + "="*60)
        print("处理完成!")
        print(f"输出文件: {output_file}")

        if original_metadata:
            print("\n已移除的metadata:")
            for key, value in original_metadata.items():
                print(f"  - {key}: {value}")
        else:
            print("\n原文件没有metadata需要移除")

        print("="*60)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
