import argparse
import os
import pysrt
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Silicon Flow API配置
SILICON_FLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3"
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

def get_total_subs(srt_files: List[str]) -> Tuple[int, dict]:
    """预计算总字幕条目数并返回有效文件列表"""
    total = 0
    valid_files = {}
    for file in srt_files:
        try:
            subs = pysrt.open(file)
            count = len(subs)
            total += count
            valid_files[file] = count
        except Exception as e:
            logging.error(f"跳过无效文件 {file}: {str(e)}")
    return total, valid_files

def process_folder(folder: str, api_key: str, max_workers: int = 8):
    """处理整个文件夹，支持并发"""
    srt_files = find_srt_files(folder)
    logging.info(f"找到 {len(srt_files)} 个待处理字幕文件")
    
    # 预计算总字幕数和有效文件
    total_subs, valid_files = get_total_subs(srt_files)
    logging.info(f"有效文件数: {len(valid_files)}")
    logging.info(f"总字幕条目数: {total_subs}")
    
    total_start = time.time()
    success_count = 0
    processed_subs = 0
    
    # 创建共享的进度条和锁
    lock = threading.Lock()
    with tqdm(total=total_subs, desc="处理字幕条目", unit="条") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_file,
                    (file, api_key, pbar, lock, valid_files[file])
                ): file
                for file in valid_files
            }
            
            for future in as_completed(futures):
                file = futures[future]
                try:
                    result, elapsed, subs_processed = future.result()
                    pbar.write(f"{result} 用时: {elapsed:.2f}秒")
                    if "成功生成" in result:
                        success_count += 1
                        processed_subs += subs_processed
                except Exception as e:
                    pbar.write(f"文件处理异常: {file}\n错误: {str(e)}")
    
    total_elapsed = time.time() - total_start
    logging.info("\n" + "=" * 50)
    logging.info(f"处理完成！")
    logging.info(f"成功处理文件数: {success_count}/{len(valid_files)}")
    logging.info(f"处理字幕条目数: {processed_subs}/{total_subs}")
    logging.info(f"总用时: {total_elapsed:.2f}秒")
    if success_count > 0:
        logging.info(f"平均速度: {processed_subs/total_elapsed:.2f} 条/秒")
    logging.info("=" * 50)

def call_silicon_flow_api(api_key: str, messages: list, temperature: float = 0.7, max_retries: int = 3) -> str:
    # 保持原有实现不变
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "max_tokens": 4096,
        "stop": ["null"],
        "temperature": temperature,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(SILICON_FLOW_API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                logging.warning(f"API请求失败，尝试重试 ({attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(2 ** attempt)  # 指数退避
            else:
                logging.error(f"API请求失败，重试次数已用完: {str(e)}")
                raise Exception(f"API请求失败: {str(e)}")
        except KeyError:
            logging.error("API响应格式异常")
            raise Exception("API响应格式异常")

def translate_srt(input_file: str, output_file: str, summary_text: str, api_key: str, pbar: tqdm, lock: threading.Lock):
    """翻译字幕文件并更新进度条"""
    try:
        subs = pysrt.open(input_file, encoding='utf-8')
    except Exception as e:
        logging.error(f"无法打开文件 {input_file}: {str(e)}")
        raise
    
    translated_subs = pysrt.SubRipFile()
    processed_count = 0
    
    for sub in subs:
        try:
            messages = [
                {"role": "system", "content": TRANLATE_SYSTEM_PROMPT.format(summary_text)},
                {"role": "user", "content": f"翻译以下字幕文本：\n{sub.text}"}
            ]
            
            translated_text = call_silicon_flow_api(
                api_key=api_key,
                messages=messages,
                temperature=0.3
            )
            
            translated_sub = pysrt.SubRipItem(
                index=sub.index,
                start=sub.start,
                end=sub.end,
                text=translated_text
            )
            translated_subs.append(translated_sub)
            processed_count += 1
        except Exception as e:
            logging.error(f"翻译字幕失败: {sub.text}, 错误: {str(e)}")
        finally:
            # 无论成功与否都更新进度条
            with lock:
                pbar.update(1)
    
    try:
        translated_subs.save(output_file, encoding='utf-8')
        return processed_count
    except Exception as e:
        logging.error(f"保存文件失败 {output_file}: {str(e)}")
        raise

def summary_srt(input_file: str, api_key: str) -> str:
    # 保持原有实现不变
    subs = pysrt.open(input_file, encoding='utf-8')
    full_text = "\n".join(sub.text for sub in subs)
    
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": f"请根据以下字幕内容生成概要：\n{full_text}"}
    ]
    
    return call_silicon_flow_api(
        api_key=api_key,
        messages=messages,
        temperature=0.7
    )

def process_single_file(args: Tuple[str, str, tqdm, threading.Lock, int]) -> Tuple[str, float, int]:
    """处理单个文件，返回处理结果、用时和处理的字幕数"""
    input_path, api_key, pbar, lock, total_subs = args
    start_time = time.time()
    processed_count = 0
    
    try:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_zh.srt"
        
        if os.path.exists(output_path):
            return (f"跳过已存在文件: {output_path}", 0, 0)
        
        summary = summary_srt(input_path, api_key)
        processed_count = translate_srt(input_path, output_path, summary, api_key, pbar, lock)
        
        elapsed = time.time() - start_time
        return (f"成功生成: {os.path.basename(output_path)}", elapsed, processed_count)
    except Exception as e:
        elapsed = time.time() - start_time
        return (f"处理失败: {input_path} ({str(e)})", elapsed, processed_count)

def main():
    # 保持原有实现不变
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
        default=4,
        help="并发处理数"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder):
        logging.error(f"错误：{args.folder} 不是有效文件夹")
        return

    api_key = os.environ.get("SILICON_FLOW_API_KEY")
    if not api_key:
        logging.error("错误：请通过环境变量 SILICON_FLOW_API_KEY 设置API密钥")
        logging.error("示例：export SILICON_FLOW_API_KEY='your-api-key'")
        return

    try:
        process_folder(args.folder, api_key, args.workers)
    except KeyboardInterrupt:
        logging.error("\n用户中断操作")

if __name__ == "__main__":
    main()