import argparse
import os
import pysrt
from ollama import Client, Options

client = Client(host="http://127.0.0.1:11434")
MODEL_NAME = "qwen2:7b-instruct-q8_0"
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


def translate_srt(input_file, output_file, summary_text):
    subs: pysrt.SubRipFile = pysrt.open(input_file, encoding='utf-8')
    translated_subs = pysrt.SubRipFile()

    for sub in subs:
        translated_text = client.generate(model=MODEL_NAME, prompt=f'翻译的文本如下所示:\n"""{sub.text}"""', system=TRANLATE_SYSTEM_PROMPT.format(
            summary_text), stream=False, options=Options(temperature=0.3, num_gpu=-1, num_thread=16,num_ctx=16384))['response']
        print(
            f"original text: {sub.text} \t translated text: {translated_text}")
        translated_sub = pysrt.SubRipItem(
            index=sub.index, start=sub.start, end=sub.end, text=translated_text)
        translated_subs.append(translated_sub)

    translated_subs.save(output_file, encoding='utf-8')


def summary_srt(input_file):
    subs = pysrt.open(input_file, encoding='utf-8')
    summary_text = ""
    for sub in subs:
        summary_text += sub.text + "\n"
    resp_text = client.generate(model=MODEL_NAME, prompt=summary_text,
                                system=SUMMARY_SYSTEM_PROMPT, stream=False, options=Options(temperature=0.7, num_gpu=-1, num_thread=16,num_ctx=16384))
    print(resp_text['response'])
    return resp_text['response']

def main():
    parser = argparse.ArgumentParser(
        description='Translate SRT subtitle files in a directory.')
    parser.add_argument('directory', type=str,
                        help='Path to the directory containing SRT subtitle files.')
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f'Error: The provided directory "{args.directory}" does not exist.')
        return

    for filename in os.listdir(args.directory):
        if filename.endswith('.srt') and not filename.endswith('_zh.srt'):
            input_file = os.path.join(args.directory, filename)
            output_file = os.path.join(args.directory, os.path.splitext(filename)[0] + '_zh.srt')

            if os.path.exists(output_file):
                print(f'Skipping {filename} as {os.path.basename(output_file)} already exists.')
                continue

            try:
                summary_text = summary_srt(input_file)
                translate_srt(input_file, output_file, summary_text)
                print(f'Translation completed for {filename}. Output saved as {os.path.basename(output_file)}.')
            except FileNotFoundError:
                print(f'Error: Could not find the input SRT file "{input_file}".')
            except pysrt.exceptions.InvalidHeader:
                print(f'Error: The input SRT file "{input_file}" has an invalid header.')
            except Exception as e:
                print(f'Error: An unexpected error occurred: {str(e)}')

if __name__ == '__main__':
    main()