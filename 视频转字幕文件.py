import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np
import os
import re
from tqdm import tqdm
import subprocess
import tempfile
import shutil
from pathlib import Path
import json
import webrtcvad  # 引入WebRTC VAD库
import collections
import contextlib
import wave
import struct

# 添加opencc用于繁简转换
import opencc

# 繁体转简体转换器
converter = opencc.OpenCC("t2s")


class Frame(object):
    """用于VAD的音频帧表示类"""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """生成音频帧"""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset : offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """使用VAD标记语音段落"""
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])

            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])

            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield voiced_frames
                ring_buffer.clear()
                voiced_frames = []

    if voiced_frames:
        yield voiced_frames


def apply_vad_to_audio(audio_file, aggressive_level=2):
    """
    应用VAD到音频文件，获取有语音的段落

    参数:
        audio_file: 音频文件路径
        aggressive_level: VAD敏感度 (0-3), 3最严格

    返回:
        list: 语音片段列表 [(start_time, end_time), ...]
    """
    print(f"使用VAD处理音频: {audio_file}, 敏感度级别: {aggressive_level}")

    # 创建临时音频文件(PCM格式，16kHz采样率，16位)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_wav = temp_file.name

    # 使用ffmpeg转换为VAD需要的格式
    command = [
        "ffmpeg",
        "-y",
        "-i",
        audio_file,
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        temp_wav,
    ]

    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 打开WAV文件
    with contextlib.closing(wave.open(temp_wav, "rb")) as wf:
        sample_rate = wf.getframerate()
        assert sample_rate in (
            8000,
            16000,
            32000,
            48000,
        ), "样本率必须是8000，16000，32000或48000"

        # 读取WAV文件内容
        pcm_data = wf.readframes(wf.getnframes())

        # 创建VAD实例
        vad = webrtcvad.Vad(aggressive_level)

        # 帧的时长 (ms)
        frame_duration = 20

        # 生成帧
        frames = list(frame_generator(frame_duration, pcm_data, sample_rate))

        # VAD处理
        segments = list(vad_collector(sample_rate, frame_duration, 300, vad, frames))

        # 转换为时间段列表
        voice_segments = []
        for segment in segments:
            if len(segment) > 0:
                start_time = segment[0].timestamp
                end_time = segment[-1].timestamp + segment[-1].duration
                voice_segments.append((start_time, end_time))

    # 清理临时文件
    os.remove(temp_wav)

    # 合并相邻的短片段
    merged_segments = []
    if voice_segments:
        current_segment = voice_segments[0]

        for segment in voice_segments[1:]:
            # 如果两个片段的间隔小于0.5秒，合并它们
            if segment[0] - current_segment[1] < 0.5:
                current_segment = (current_segment[0], segment[1])
            else:
                merged_segments.append(current_segment)
                current_segment = segment

        merged_segments.append(current_segment)

    print(f"VAD识别出 {len(merged_segments)} 个语音段落")

    return merged_segments


def generate_subtitle(
    audio_file,
    output_srt=None,
    model_size="large",  # 更改为large模型
    language="zh",
    segment_length=30,
    overlap=2,
    use_vad=True,  # 启用VAD
    vad_level=2,  # VAD敏感度
    silence_threshold=0.025,
    min_silence_duration=0.5,
    precision_mode="high",
):
    """
    将音频文件转换为高精度的 SRT 字幕文件，通过分段处理长音频

    参数:
        audio_file (str): 输入音频文件路径
        output_srt (str, optional): 输出 SRT 文件路径
        model_size (str, optional): Whisper 模型大小 ("tiny", "base", "small", "medium", "large")
        language (str, optional): 音频语言代码，默认为中文 "zh"
        segment_length (int, optional): 音频分段长度（秒）
        overlap (int, optional): 分段重叠时间（秒）
        use_vad (bool, optional): 是否使用VAD过滤静音
        vad_level (int, optional): VAD敏感度 (0-3)
        silence_threshold (float, optional): 静音检测阈值 (0-1)
        min_silence_duration (float, optional): 最小静音持续时间（秒）
        precision_mode (str, optional): 精度模式 ("standard", "high", "maximum")

    返回:
        str: 生成的字幕文件路径
        str: 生成的完整转录文本
    """
    # 设置默认输出文件名
    if output_srt is None:
        base_name = os.path.splitext(audio_file)[0]
        output_srt = f"{base_name}.srt"

    print(f"处理音频文件: {audio_file}")

    # 加载音频
    print("加载和分析音频...")
    input_audio, sampling_rate = librosa.load(audio_file, sr=16000)

    # 计算音频时长
    audio_duration = len(input_audio) / sampling_rate
    print(f"音频时长: {audio_duration:.2f} 秒")

    # 使用VAD识别语音段落
    voice_segments = []
    if use_vad:
        print("使用VAD检测语音活动...")
        voice_segments = apply_vad_to_audio(audio_file, aggressive_level=vad_level)

        if not voice_segments:
            print("VAD未检测到有效语音，将使用整个音频进行处理")
            voice_segments = [(0, audio_duration)]
    else:
        voice_segments = [(0, audio_duration)]

    # 加载模型和处理器
    print(f"加载Whisper {model_size}模型...")
    model_id = f"openai/whisper-{model_size}"
    processor = WhisperProcessor.from_pretrained(model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )

    # 存储所有分段的转录结果，带时间戳
    all_segments = []
    full_transcription = ""

    # 处理每个识别出的语音段落
    for segment_idx, (segment_start, segment_end) in enumerate(voice_segments):
        print(
            f"处理语音段落 {segment_idx+1}/{len(voice_segments)} ({segment_start:.2f}s - {segment_end:.2f}s)"
        )

        # 计算对应的音频样本位置
        start_sample = int(segment_start * sampling_rate)
        end_sample = int(segment_end * sampling_rate)

        # 提取当前语音段落音频
        segment_audio = input_audio[start_sample:end_sample]
        segment_duration = len(segment_audio) / sampling_rate

        # 如果段落太长，进一步分割处理
        if segment_duration > segment_length:
            subsegment_samples = segment_length * sampling_rate
            overlap_samples = overlap * sampling_rate

            # 计算子段落数量
            num_subsegments = max(
                1,
                int(
                    np.ceil(len(segment_audio) / (subsegment_samples - overlap_samples))
                ),
            )

            segment_results = []

            # 处理每个子段落
            for i in range(num_subsegments):
                start_sample_rel = max(0, i * (subsegment_samples - overlap_samples))
                end_sample_rel = min(
                    len(segment_audio), start_sample_rel + subsegment_samples
                )

                # 计算相对时间戳（秒）
                subsegment_start = segment_start + (start_sample_rel / sampling_rate)
                subsegment_end = segment_start + (end_sample_rel / sampling_rate)

                # 提取当前子段落
                subsegment_audio = segment_audio[start_sample_rel:end_sample_rel]

                print(
                    f"  - 处理子段落 {i+1}/{num_subsegments} ({subsegment_start:.2f}s - {subsegment_end:.2f}s)"
                )

                # 转录当前子段落
                transcription = transcribe_audio_segment(
                    subsegment_audio, sampling_rate, model, processor, device, language
                )

                if transcription.strip():
                    segment_results.append(
                        (subsegment_start, subsegment_end, transcription)
                    )

            # 合并子段落转录结果
            if segment_results:
                merged_transcription = smart_merge_transcriptions(
                    [t for _, _, t in segment_results]
                )

                # 使用整个段落的时间戳
                all_segments.append((segment_start, segment_end, merged_transcription))
                full_transcription += merged_transcription + " "
        else:
            # 段落长度合适，直接处理
            transcription = transcribe_audio_segment(
                segment_audio, sampling_rate, model, processor, device, language
            )

            if transcription.strip():
                all_segments.append((segment_start, segment_end, transcription))
                full_transcription += transcription + " "

    # 清理并整理完整转录
    full_transcription = full_transcription.strip()
    print(
        f"完整转录: {full_transcription[:100]}..."
        if len(full_transcription) > 100
        else f"完整转录: {full_transcription}"
    )

    # 生成最终的字幕分段
    subtitle_segments = []

    # 基于完整转录和时间戳创建字幕
    for start, end, text in all_segments:
        # 将文本分成自然句子
        sentences = semantic_split(text)

        if len(sentences) == 1:
            # 如果只有一个句子，直接使用原始时间戳
            subtitle_segments.append({"start": start, "end": end, "text": sentences[0]})
        else:
            # 多个句子，分配时间戳
            segment_duration = end - start
            char_count = sum(len(s) for s in sentences)

            current_time = start
            for sentence in sentences:
                if char_count > 0:
                    # 根据字符数分配时长
                    duration = (len(sentence) / char_count) * segment_duration
                    sentence_end = min(current_time + duration, end)

                    subtitle_segments.append(
                        {"start": current_time, "end": sentence_end, "text": sentence}
                    )

                    current_time = sentence_end

    print(f"生成了 {len(subtitle_segments)} 个字幕分段")

    # 写入 SRT 文件
    with open(output_srt, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(subtitle_segments):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()

            # 确保文本是简体中文
            text = converter.convert(text)

            # 写入 SRT 格式
            srt_file.write(f"{i+1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")

    print(f"字幕文件已保存至: {output_srt}")

    return output_srt, full_transcription


def transcribe_audio_segment(
    audio_segment, sampling_rate, model, processor, device, language
):
    """转录单个音频段落"""
    # 准备音频特征
    input_features = processor.feature_extractor(
        audio_segment, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features

    input_features = input_features.to(device=device, dtype=torch.float16)

    # 设置解码参数
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )

    with torch.no_grad():
        generated_ids = model.generate(
            input_features=input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_length=448,
            temperature=0.0,
        )

    # 解码转录文本
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 繁体转简体
    transcription = converter.convert(transcription)

    return transcription


def smart_merge_transcriptions(transcriptions):
    """
    智能合并多个转录片段，处理重叠部分

    参数:
        transcriptions (list): 转录片段列表

    返回:
        str: 合并后的完整转录
    """
    if not transcriptions:
        return ""

    if len(transcriptions) == 1:
        return transcriptions[0]

    # 合并所有转录，处理重叠部分
    merged = transcriptions[0]

    for i in range(1, len(transcriptions)):
        current = transcriptions[i]
        previous = merged

        # 寻找重叠部分（至少3个字符）
        overlap_length = min(len(previous), len(current))
        max_overlap = 0

        for j in range(3, min(20, overlap_length)):  # 限制搜索范围，避免过度匹配
            # 检查前一个转录的末尾与当前转录的开头是否匹配
            if previous[-j:] == current[:j]:
                max_overlap = j

        if max_overlap > 0:
            # 找到重叠，从重叠部分开始合并
            merged = previous + current[max_overlap:]
        else:
            # 没有明显重叠，添加空格分隔
            merged = previous + " " + current

    return merged


def semantic_split(text):
    """基于语义边界分割文本"""
    # 使用句子级别的标点符号分割
    pattern = r"([。！？；!?;]+)"
    parts = re.split(pattern, text)

    sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts):
            sentence = parts[i] + parts[i + 1]
            sentences.append(sentence)
            i += 2
        else:
            if parts[i].strip():
                sentences.append(parts[i])
            i += 1

    # 如果没有足够的句子，尝试使用逗号分割
    if len(sentences) <= 3:
        sentences = []
        pattern = r"([，,、]+)"
        parts = re.split(pattern, text)

        i = 0
        while i < len(parts):
            if i + 1 < len(parts):
                sentence = parts[i] + parts[i + 1]
                sentences.append(sentence)
                i += 2
            else:
                if parts[i].strip():
                    sentences.append(parts[i])
                i += 1

    return [s.strip() for s in sentences if s.strip()]


def format_timestamp(seconds):
    """将秒数转换为 SRT 格式的时间戳：00:00:00,000"""
    milliseconds = int((seconds % 1) * 1000)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


# 以下是视频处理相关函数
def check_ffmpeg():
    """检查ffmpeg是否可用"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_audio_from_video(video_path):
    """从视频文件中提取音频"""
    temp_dir = tempfile.mkdtemp()
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_audio = os.path.join(temp_dir, f"{base_name}.wav")

    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-q:a",
        "0",
        "-map",
        "a",
        "-vn",
        output_audio,
    ]

    try:
        subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(f"从 {video_path} 提取音频到 {output_audio}")
        return output_audio
    except subprocess.CalledProcessError as e:
        print(f"提取音频失败: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None


def is_video_file(file_path):
    """检查文件是否为视频文件"""
    video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"]
    _, ext = os.path.splitext(file_path.lower())
    return ext in video_extensions


def process_video(video_file, language="zh", model_size="large", course_data=None):
    """处理单个视频文件"""
    try:
        print(f"\n===== 开始处理视频: {video_file} =====")

        # 获取视频标题（文件名）
        video_title = os.path.splitext(os.path.basename(video_file))[0]

        # 提取音频
        audio_file = extract_audio_from_video(video_file)
        if not audio_file:
            print(f"无法从视频中提取音频: {video_file}")
            return False

        # 生成字幕文件
        subtitle_path = os.path.splitext(video_file)[0] + ".srt"
        _, transcription = generate_subtitle(
            audio_file,
            subtitle_path,
            model_size=model_size,
            language=language,
            segment_length=30,
            overlap=2,
            use_vad=True,  # 启用VAD
            vad_level=2,  # 默认VAD敏感度
        )

        # 确保转录文本是简体中文
        transcription = converter.convert(transcription)

        # 将转录文本添加到课程数据中
        if course_data is not None:
            course_data.append({"title": video_title, "text": [transcription]})

        # 清理临时音频文件
        temp_dir = os.path.dirname(audio_file)
        shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"===== 视频处理完成: {video_file} =====\n")
        return True
    except Exception as e:
        print(f"处理视频 {video_file} 时出错: {str(e)}")
        return False


def process_videos_in_directory(directory_path, language="zh", model_size="large"):
    """处理目录中的所有视频文件，生成字幕"""
    # 确保目录存在
    if not os.path.exists(directory_path):
        print(f"目录不存在: {directory_path}")
        return []

    # 检查ffmpeg是否可用
    if not check_ffmpeg():
        print("错误: ffmpeg未安装或不在系统路径中。请安装ffmpeg后再试。")
        return []

    # 递归获取所有视频文件
    video_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if is_video_file(file_path):
                video_files.append(file_path)

    if not video_files:
        print(f"在 {directory_path} 中没有找到视频文件")
        return []

    print(f"找到 {len(video_files)} 个视频文件")

    # 准备课程数据列表
    course_data = []

    # 处理每个视频文件
    success_count = 0
    for i, video_file in enumerate(video_files):
        print(f"处理进度: [{i+1}/{len(video_files)}]")
        if process_video(video_file, language, model_size, course_data):
            success_count += 1

    print(f"处理完成! 成功: {success_count}/{len(video_files)}")

    # 保存课程数据到JSON文件
    output_json = os.path.join(directory_path, "course_data.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(course_data, f, ensure_ascii=False, indent=4)

    print(f"课程数据已保存至: {output_json}")

    return course_data


def main(directory_path, language="zh", model_size="large"):
    """主函数"""
    print("=" * 50)
    print(f"视频字幕生成工具")
    print("=" * 50)
    print(f"处理目录: {directory_path}")
    print(f"语言设置: {language}")
    print(f"模型大小: {model_size}")
    print("=" * 50)

    # 转换为绝对路径
    directory_path = os.path.abspath(directory_path)

    # 处理目录中的所有视频
    course_data = process_videos_in_directory(directory_path, language, model_size)

    # 如果没有成功处理任何视频，返回
    if not course_data:
        print("没有成功处理任何视频，未生成JSON文件。")
        return

    print(f"处理完成！共生成 {len(course_data)} 个课程数据。")


if __name__ == "__main__":
    # 在这里直接指定要处理的目录路径
    # 如果想要处理不同的目录，请修改这里的路径
    main("")  # 请根据实际情况修改此路径
