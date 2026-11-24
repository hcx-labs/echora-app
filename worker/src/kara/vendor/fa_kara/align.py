import torch
import torchaudio
import librosa
import math
import numpy as np
import time

def align_audio_with_text(audio_file_path, text_tokens, non_silent_ranges=[], sr=None, speed=1, use_gpu=True):
    start_time = time.time()
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    try:
        bundle = torchaudio.pipelines.MMS_FA
        if isinstance(audio_file_path, str):
            waveform, sample_rate = torchaudio.load(audio_file_path)
        else:
            waveform = torch.tensor(audio_file_path).float()
            waveform = waveform.unsqueeze(0)
            sample_rate = sr
        
        # 处理非静音区域
        if non_silent_ranges:
            # 将时间(秒)转换为样本点
            total_samples = waveform.shape[1]
            sample_ranges = []
            for start_sec, end_sec in non_silent_ranges:
                start_sample = int(start_sec * sample_rate / speed)
                end_sample = min(int(end_sec * sample_rate / speed), total_samples)
                sample_ranges.append((start_sample, end_sample))
            
            # 提取并拼接非静音片段
            segments = []
            for start, end in sample_ranges:
                segments.append(waveform[:, start:end])
            waveform = torch.cat(segments, dim=1)
        
        # 单声道处理
        waveform = waveform.mean(0, keepdim=True)
        
        # 重采样
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, bundle.sample_rate
        )
        
        # 加载模型
        model = bundle.get_model().to(device)
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()
        
        # 处理有效token
        valid_tokens = [token for token in text_tokens if token]
        
        with torch.inference_mode():
            emission, _ = model(waveform.to(device))
            tokens = tokenizer(valid_tokens)
            token_spans = aligner(emission[0], tokens)
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 时间转换参数
        frame_duration = 1.0 / bundle.sample_rate * 320 * speed
        results = []
        
        # 映射回原始时间
        def map_to_original_time(adjusted_time):
            """将处理后的时间映射回原始音频时间"""
            if not non_silent_ranges:
                return adjusted_time
            
            cumulative_duration = 0.0
            for start_sec, end_sec in non_silent_ranges:
                segment_duration = end_sec - start_sec
                if adjusted_time < cumulative_duration + segment_duration:
                    return start_sec + (adjusted_time - cumulative_duration)
                cumulative_duration += segment_duration
            return non_silent_ranges[-1][1]  # 超出范围返回最后时间
        
        # 时间格式化函数
        def format_time(time_sec):
            minutes, remainder = divmod(time_sec, 60)
            seconds, centiseconds = divmod(remainder, 1)
            return f"[{int(minutes):02d}:{int(seconds):02d}:{math.floor(centiseconds * 100):02d}]"

        # 处理每个token的时间对齐
        for i, spans in enumerate(token_spans):
            if not spans:
                results.append({
                    'token': valid_tokens[i],
                    'start': '[error]',
                    'end': '[error]'
                })
                continue
                
            # 获取调整后的时间
            adjusted_start = spans[0].start * frame_duration
            adjusted_end = spans[-1].end * frame_duration
            
            # 映射回原始音频时间
            original_start = map_to_original_time(adjusted_start)
            original_end = map_to_original_time(adjusted_end)
            
            results.append({
                'token': valid_tokens[i],
                'start': format_time(original_start),
                'end': format_time(original_end),
                'original_start': original_start,
                'original_end': original_end
            })
        
        end_time = time.time()
        print("Alignment inference executed in", round(end_time - start_time, 3), "seconds")
        return results

    except Exception as e:
        print(f"Error during alignment: {e}")
        return []


def align_audio_with_text_torch_yohane(audio_file_path, text_tokens, non_silent_ranges=[], sr=None, speed=1, use_gpu=True):
    ' 使用 yohane 的 TorchAudioForcedAligner 类复现 '

    from align_yohane import TorchAudioForcedAligner

    start_time = time.time()
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    try:
        if isinstance(audio_file_path, str):
            waveform, sample_rate = torchaudio.load(audio_file_path)
        else:
            waveform = torch.tensor(audio_file_path).float()
            waveform = waveform.unsqueeze(0)
            sample_rate = sr
        
        # 处理非静音区域
        if non_silent_ranges:
            # 将时间(秒)转换为样本点
            total_samples = waveform.shape[1]
            sample_ranges = []
            for start_sec, end_sec in non_silent_ranges:
                start_sample = int(start_sec * sample_rate / speed)
                end_sample = min(int(end_sec * sample_rate / speed), total_samples)
                sample_ranges.append((start_sample, end_sample))
            
            # 提取并拼接非静音片段
            segments = []
            for start, end in sample_ranges:
                segments.append(waveform[:, start:end])
            waveform = torch.cat(segments, dim=1)

        # 处理有效token
        valid_tokens = [token for token in text_tokens if token]

        # from yohane-2026.5.0
        torch_aligner = TorchAudioForcedAligner()
        tokens = torch_aligner.tokenize(valid_tokens)
        _, token_spans, tgt_sample_rate = torch_aligner.align(tokens, waveform, sample_rate)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 时间转换参数
        frame_duration = 1.0 / tgt_sample_rate * 320 * speed
        results = []
        
        # 映射回原始时间
        def map_to_original_time(adjusted_time):
            """将处理后的时间映射回原始音频时间"""
            if not non_silent_ranges:
                return adjusted_time
            
            cumulative_duration = 0.0
            for start_sec, end_sec in non_silent_ranges:
                segment_duration = end_sec - start_sec
                if adjusted_time < cumulative_duration + segment_duration:
                    return start_sec + (adjusted_time - cumulative_duration)
                cumulative_duration += segment_duration
            return non_silent_ranges[-1][1]  # 超出范围返回最后时间
        
        # 时间格式化函数
        def format_time(time_sec):
            minutes, remainder = divmod(time_sec, 60)
            seconds, centiseconds = divmod(remainder, 1)
            return f"[{int(minutes):02d}:{int(seconds):02d}:{math.floor(centiseconds * 100):02d}]"

        # 处理每个token的时间对齐
        for i, spans in enumerate(token_spans):
            if not spans:
                results.append({
                    'token': valid_tokens[i],
                    'start': '[error]',
                    'end': '[error]'
                })
                continue
                
            # 获取调整后的时间
            adjusted_start = spans[0].start * frame_duration
            adjusted_end = spans[-1].end * frame_duration
            
            # 映射回原始音频时间
            original_start = map_to_original_time(adjusted_start)
            original_end = map_to_original_time(adjusted_end)
            
            results.append({
                'token': valid_tokens[i],
                'start': format_time(original_start),
                'end': format_time(original_end),
                'original_start': original_start,
                'original_end': original_end
            })
        
        end_time = time.time()
        print("Alignment inference executed in", round(end_time - start_time, 3), "seconds")
        return results

    except Exception as e:
        print(f"Error during alignment: {e}")
        return []