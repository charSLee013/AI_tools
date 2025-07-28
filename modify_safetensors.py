#!/usr/bin/env python3
"""
SafeTensors文件修改脚本
以最小代价修改safetensors文件的MD5哈希值
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

def analyze_and_modify_safetensors(input_file, output_file, modification_value=1e-8):
    """
    分析safetensors文件并进行最小修改
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        modification_value: 修改值大小，默认1e-8
    """
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"修改值: {modification_value}")
    print("-" * 60)
    
    # 计算原始文件MD5
    original_md5 = get_file_md5(input_file)
    print(f"原始MD5: {original_md5}")
    
    # 加载所有张量
    print("加载张量数据...")
    tensors = load_file(input_file)
    
    # 获取张量信息
    tensor_names = list(tensors.keys())
    print(f"总张量数量: {len(tensor_names)}")
    
    # 分析最后一个张量
    last_tensor_name = tensor_names[-1]
    last_tensor = tensors[last_tensor_name]
    
    print(f"\n目标张量信息:")
    print(f"  名称: {last_tensor_name}")
    print(f"  形状: {list(last_tensor.shape)}")
    print(f"  类型: {last_tensor.dtype}")
    print(f"  元素数: {last_tensor.numel()}")
    
    # 显示修改前的最后几个元素
    flat_tensor = last_tensor.flatten()
    last_elements = flat_tensor[-5:] if flat_tensor.numel() >= 5 else flat_tensor
    print(f"  修改前最后5个元素: {[f'{x.item():.8f}' for x in last_elements]}")
    
    # 进行最小修改：修改最后一个元素
    print(f"\n执行修改...")
    modified_tensor = last_tensor.clone()
    
    # 获取最后一个元素的索引
    if len(modified_tensor.shape) == 1:
        # 1D张量
        original_value = modified_tensor[-1].item()
        modified_tensor[-1] += modification_value
        new_value = modified_tensor[-1].item()
    elif len(modified_tensor.shape) == 2:
        # 2D张量
        original_value = modified_tensor[-1, -1].item()
        modified_tensor[-1, -1] += modification_value
        new_value = modified_tensor[-1, -1].item()
    elif len(modified_tensor.shape) == 3:
        # 3D张量
        original_value = modified_tensor[-1, -1, -1].item()
        modified_tensor[-1, -1, -1] += modification_value
        new_value = modified_tensor[-1, -1, -1].item()
    elif len(modified_tensor.shape) == 4:
        # 4D张量
        original_value = modified_tensor[-1, -1, -1, -1].item()
        modified_tensor[-1, -1, -1, -1] += modification_value
        new_value = modified_tensor[-1, -1, -1, -1].item()
    else:
        # 多维张量，使用flatten方式
        flat_modified = modified_tensor.flatten()
        original_value = flat_modified[-1].item()
        flat_modified[-1] += modification_value
        new_value = flat_modified[-1].item()
        modified_tensor = flat_modified.reshape(modified_tensor.shape)
    
    print(f"  最后元素原值: {original_value:.8f}")
    print(f"  最后元素新值: {new_value:.8f}")
    print(f"  修改差值: {new_value - original_value:.8f}")
    
    # 更新张量字典
    tensors[last_tensor_name] = modified_tensor
    
    # 显示修改后的最后几个元素
    flat_tensor_new = modified_tensor.flatten()
    last_elements_new = flat_tensor_new[-5:] if flat_tensor_new.numel() >= 5 else flat_tensor_new
    print(f"  修改后最后5个元素: {[f'{x.item():.8f}' for x in last_elements_new]}")
    
    # 保存修改后的文件
    print(f"\n保存修改后的文件...")
    save_file(tensors, output_file)
    
    # 验证MD5变化
    new_md5 = get_file_md5(output_file)
    print(f"\n验证结果:")
    print(f"原始MD5: {original_md5}")
    print(f"新文件MD5: {new_md5}")
    print(f"MD5已改变: {'是' if original_md5 != new_md5 else '否'}")
    
    # 文件大小对比
    original_size = os.path.getsize(input_file)
    new_size = os.path.getsize(output_file)
    print(f"原始文件大小: {original_size / (1024*1024):.2f} MB")
    print(f"新文件大小: {new_size / (1024*1024):.2f} MB")
    print(f"大小差异: {abs(new_size - original_size)} 字节")
    
    return original_md5, new_md5

def main():
    if len(sys.argv) < 2:
        print("用法: python modify_safetensors.py <输入文件> [输出文件] [修改值]")
        print("示例: python modify_safetensors.py realcomic_000000900.safetensors")
        print("示例: python modify_safetensors.py input.safetensors output.safetensors 1e-10")
        exit(1)
    
    input_file = sys.argv[1]
    
    # 生成默认输出文件名
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_modified.safetensors"
    
    # 修改值
    modification_value = float(sys.argv[3]) if len(sys.argv) >= 4 else 1e-8
    
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
        print("SafeTensors MD5修改工具")
        print("="*60)
        
        original_md5, new_md5 = analyze_and_modify_safetensors(
            input_file, output_file, modification_value
        )
        
        print("\n" + "="*60)
        print("修改完成!")
        print(f"输出文件: {output_file}")
        print(f"MD5变化: {original_md5} -> {new_md5}")
        print("="*60)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
