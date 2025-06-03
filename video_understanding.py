import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import time
# os.environ['https_proxy'] = 'http://gzbh-aip-paddlecloud140.gzbh:8128'

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

import os
import hashlib
import requests

from IPython.display import Markdown, display
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu


def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")


def get_video_frames(video_path, num_frames=128, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)
    
    return video_file_path, frames, timestamps


def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = (len(images) + num_columns - 1) // num_columns

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image


def inference(video_path, prompt, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]


video_url = "test_video/小林说-碳排放-Scene-001.mp4"
# prompt = "请用表格总结一下视频中的商品特点"
# prompt = "请描述一下视频的内容摘要"
# prompt = "我想分析视频的剪辑制作手法，请描述画面使用的素材类型和空间布局,比如：“该视频背景为虚化模糊效果，画面中有三个画中画，中间两个窗口分别展示视频素材和图表，右下角放置主播讲解的窗口”"
prompt = """
作为专业视频制作分析师，请按以下步骤解析当前视频片段：

画面元素解构

素材类型：识别所有视觉元素（真人出镜口播/视频素材/图像素材/动画/3D模型/文字/图表等）

空间拓扑：使用坐标系描述元素位置（示例：x轴0-100，y轴0-100）

空间占比：每一类内容占整个画面的比例

层级关系：标注元素层级（背景层/主视觉层/叠加层）

动态效果分析

运镜方式：推/拉/摇/移/跟（标注幅度和速度）

元素动态：入场/退场动画类型（缩放/平移/旋转等）

转场技巧：若为场景切换点，标注转场类型

专业技法识别

色彩工程：主色调/对比色/渐变应用

视觉引导：视觉焦点转移路径（Z型/S型等构图）

隐藏设计解析

注意力引导：突出核心信息的视觉策略

情感渲染：色调/运镜与内容情感的协同

信息密度：单位时间内视觉信息量评估

美学评价

排版

吸引力

请使用如下格式结构化输出：
【画面解构】
素材构成：[元素列表]
空间拓扑：[坐标系定位]
空间占比：[每种元素占据画面的比例]
景深结构：[前景/中景/背景描述]

【动态描述】
运镜模式：[技术术语+参数]
动态语法：[关键帧变化描述]
视觉节奏：[镜头时长与动作匹配度]

【视频设计思路】
核心焦点：[视觉焦点转移路径]
认知负荷：[信息吸收难易度评估]
导演意图：[推测的内容表达策略]"

【美学评价】
配色：[配色思路/评分]
布局：[布局思路/评分]
美学：[美学评价/评分]
"""
video_path, frames, timestamps = get_video_frames(video_url, num_frames=64)
# image_grid = create_image_grid(frames, num_columns=8)
# display(image_grid.resize((640, 640)))
s = time.time()
response = inference(video_path, prompt)
print(response)
e = time.time()
print("time:", e - s)
# display(Markdown(response))
