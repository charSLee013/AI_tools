import argparse
import os
import pysrt
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

# DeepSeek API配置
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-chat"
TRANLATE_SYSTEM_PROMPT = """你是一名专业的翻译官。你的任务是将用户输入的视频里面每句话都翻译成[中文]。如果遇到专业术语，请保持原样不变。确保翻译考虑到字幕的上下文，以准确传达原意。不要说对不起或者其他话语，也不要说抱歉和多余的其他话。可以结合下面的视频概要理解所要翻译的具体内容:
{}
"""

SUMMARY_SYSTEM_PROMPT = """## System Prompt for Generating Brief Summary

Identity: You are an advanced language model trained to extract and condense key information from extensive texts into a concise summary.

Task: Generate a brief summary that encapsulates the essence and key elements of the provided text, focusing on the main points and central themes.

Guidelines for Summary:
- Extract the most significant and relevant information that represents the core of the text.
- Focus on the main ideas, key findings, or unique aspects mentioned in the text.
- Keep the summary concise and to the point, avoiding any unnecessary details or elaborations.

Please ensure the summary is succinct and captures the main essence of the text without diverging into less relevant details.
"""


def find_srt_files(folder: str) -> List[str]:
    """查找需要处理的SRT文件"""
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".srt") 
        and not f.endswith("_zh.srt")
        and os.path.isfile(os.path.join(folder, f))
    ]

def process_folder(folder: str, api_key: str):
    """处理整个文件夹"""
    srt_files = find_srt_files(folder)
    print(f"找到 {len(srt_files)} 个待处理字幕文件")
    
    for idx, input_path in enumerate(srt_files, 1):
        try:
            print(f"\n正在处理文件 ({idx}/{len(srt_files)}): {os.path.basename(input_path)}")
            
            # 生成输出路径
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_zh.srt"
            
            if os.path.exists(output_path):
                print(f"跳过已存在文件: {output_path}")
                continue

            # 执行处理流程
            summary = summary_srt(input_path, api_key)
            translate_srt(input_path, output_path, summary, api_key)
            
            print(f"成功生成: {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"处理文件失败: {input_path}\n错误信息: {str(e)}")

def call_deepseek_api(api_key: str, messages: list, temperature: float = 0.7) -> str:
    """
    封装DeepSeek API调用
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        # "temperature": temperature,
        "stream": False
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"API请求失败: {str(e)}")
    except KeyError:
        raise Exception("API响应格式异常")

def translate_srt(input_file: str, output_file: str, summary_text: str, api_key: str):
    subs = pysrt.open(input_file, encoding='utf-8')
    translated_subs = pysrt.SubRipFile()

    for sub in subs:
        # 构建对话消息
        messages = [
            {"role": "system", "content": TRANLATE_SYSTEM_PROMPT.format(summary_text)},
            {"role": "user", "content": f"翻译以下字幕文本：\n{sub.text}"}
        ]
        
        translated_text = call_deepseek_api(
            api_key=api_key,
            messages=messages,
            temperature=0.3  # 确定性较高的翻译
        )
        
        # print(f"原文: {sub.text} \n译文: {translated_text}\n{'-'*50}")
        translated_sub = pysrt.SubRipItem(
            index=sub.index,
            start=sub.start,
            end=sub.end,
            text=translated_text
        )
        translated_subs.append(translated_sub)

    translated_subs.save(output_file, encoding='utf-8')

def summary_srt(input_file: str, api_key: str) -> str:
    subs = pysrt.open(input_file, encoding='utf-8')
    full_text = "\n".join(sub.text for sub in subs)
    
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": f"请根据以下字幕内容生成概要：\n{full_text}"}
    ]
    
    return call_deepseek_api(
        api_key=api_key,
        messages=messages,
        temperature=0.7  # 更具创造性的摘要
    )


def process_single_file(args: Tuple[str, str]) -> Tuple[str, float]:
    """处理单个文件，返回文件名和处理时间"""
    input_path, api_key = args
    start_time = time.time()
    
    try:
        # 生成输出路径
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_zh.srt"
        
        if os.path.exists(output_path):
            return (f"跳过已存在文件: {output_path}", 0)
        
        # 执行处理流程
        summary = summary_srt(input_path, api_key)
        translate_srt(input_path, output_path, summary, api_key)
        
        elapsed = time.time() - start_time
        return (f"成功生成: {os.path.basename(output_path)}", elapsed)
        
    except Exception as e:
        elapsed = time.time() - start_time
        return (f"处理失败: {input_path} ({str(e)})", elapsed)

def process_folder(folder: str, api_key: str, max_workers: int = 8):
    """处理整个文件夹，支持并发"""
    srt_files = find_srt_files(folder)
    print(f"找到 {len(srt_files)} 个待处理字幕文件")
    
    total_start = time.time()
    success_count = 0
    total_time = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_file, (file, api_key)): file
            for file in srt_files
        }
        
        for future in as_completed(futures):
            file = futures[future]
            try:
                result, elapsed = future.result()
                print(f"\n{result} 用时: {elapsed:.2f}秒")
                if "成功生成" in result:
                    success_count += 1
                    total_time += elapsed
            except Exception as e:
                print(f"\n文件处理异常: {file}\n错误: {str(e)}")
    
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 50)
    print(f"处理完成！")
    print(f"成功处理文件数: {success_count}/{len(srt_files)}")
    print(f"总用时: {total_elapsed:.2f}秒")
    if success_count > 0:
        print(f"平均每个文件用时: {total_time/success_count:.2f}秒")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(
        description="批量字幕翻译工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "folder",
        type=str,
        help="包含SRT字幕文件的文件夹路径"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="并发处理数"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder):
        print(f"错误：{args.folder} 不是有效文件夹")
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：请通过环境变量 DEEPSEEK_API_KEY 设置API密钥")
        print("示例：export DEEPSEEK_API_KEY='your-api-key'")
        return

    try:
        start_time = time.time()
        process_folder(args.folder, api_key, args.workers)
        print(f"\n总运行时间: {time.time() - start_time:.2f}秒")
    except KeyboardInterrupt:
        print("\n用户中断操作")

if __name__ == "__main__":
    main()