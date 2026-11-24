import bisect
import librosa
import numpy as np
import time

from utils_basic import parse_time_to_hundredths, format_hundredths_to_time_str

def non_silent_recog(audio_file, sr = None, frame_second = 1, threspct = 10, thresrto = .1):
    '识别非静音片段'
    frame_length = int(sr * frame_second)
    hop_length = frame_length // 2  # 50% 重叠
    energy = librosa.feature.rms(y=audio_file, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = np.percentile(energy, 100-threspct) * thresrto
    non_silent_frames = energy > threshold
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length) # 转换为时间点
    segments = [] # 合并连续片段
    start = None
    for i, (t, active) in enumerate(zip(times, non_silent_frames)):
        if active and start is None:
            start = max(t-frame_second/4, 0)
        elif not active and start is not None:
            segments.append((start, t+frame_second/4))
            start = None
    if start is not None:
        segments.append((start, times[-1]))
    return segments

def time_stretch_audio(audio: np.ndarray, speed: float, verbose: bool = True) -> np.ndarray:
    """
    对音频进行时间拉伸（变速）处理。

    Args:
        audio (np.ndarray): 原始音频信号（一维数组）。
        speed (float): 变速速率，>1 为加速（时长缩短），<1 为减速（时长增加），1 表示不变。
        verbose (bool): 是否打印运行时间和状态信息，默认为 True。

    Returns:
        np.ndarray: 变速后的音频信号。如果 speed == 1，则返回原始音频。
    """
    if speed == 1:
        return audio

    if verbose:
        print('Changing the audio speed...')
        start_time = time.time()

    y_processed = librosa.effects.time_stretch(audio, rate=speed)

    if verbose:
        end_time = time.time()
        print(f"Audio speed changing executed in {round(end_time - start_time, 3)} seconds")

    return y_processed

def func_tail_correct_v250615(result_list, audio_file, sr, tail_thres_pct, tail_thres_ratio, frame_second=.02):
    '新版尾音调整后处理函数'
    if not result_list:
        return []
    ns_small = non_silent_recog(audio_file, sr, frame_second, tail_thres_pct, tail_thres_ratio)
    ns_ends = [int(np.ceil(ns_end * 100)) for _, ns_end in ns_small]
    for i in range(len(result_list)-1):
        if result_list[i]['type'] != 0 and result_list[i+1]['type'] == 0:
            current_end = parse_time_to_hundredths(result_list[i]['end'])
            next_ind = i + 2
            next_start = np.inf
            while next_ind < len(result_list):
                if 'start' in result_list[next_ind]:
                    next_start = parse_time_to_hundredths(result_list[next_ind]['start'])
                    break
                next_ind += 1
            left_index = bisect.bisect_left(ns_ends, current_end)
            right_index = bisect.bisect_left(ns_ends, next_start)
            if left_index < right_index and left_index < len(ns_ends):
                result_list[i]['end'] = format_hundredths_to_time_str(ns_ends[left_index])
            else:
                interval_covered = False # 检查非静音段是否覆盖整个区间
                for nss_start, nss_end in ns_small:
                    if int(nss_start * 100) > current_end:
                        break
                    if int(nss_start * 100) <= current_end and int(np.ceil(nss_end * 100)) >= next_start:
                        interval_covered = True
                        break
                if interval_covered:
                    result_list[i]['end'] = format_hundredths_to_time_str(max(next_start-2, current_end))
    return result_list