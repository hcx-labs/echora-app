import numpy as np
import re
import unicodedata

def parse_time_to_hundredths(time_str: str) -> int:
    '.lrc格式时轴转换为厘秒整数'
    match = re.match(r'\[(\d{2}):(\d{2}):(\d{2})\]', time_str)
    minutes, seconds, hundredths = int(match.group(1)), int(match.group(2)), int(match.group(3))
    return minutes * 6000 + seconds * 100 + hundredths

def format_hundredths_to_time_str(total_hundredths: int) -> str:
    '厘秒整数转换为.lrc格式时轴'
    total_hundredths = round(total_hundredths)
    minutes = total_hundredths // 6000
    remaining = total_hundredths % 6000
    seconds = remaining // 100
    cs = remaining % 100
    return f"[{minutes:02d}:{seconds:02d}:{cs:02d}]"

def int2asstime(cs: int) -> str:
    '厘秒整数转换为.ass格式时轴'
    cs = round(cs)
    hours = cs // 360000
    cs %= 360000
    minutes = cs // 6000
    cs %= 6000
    seconds = cs // 100
    cs %= 100
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"

def calculate_length(surface):
    "结合全半角计算字符串的长度"
    length = 0.0
    for char in surface:
        if unicodedata.east_asian_width(char) in ('F', 'W', 'A'):
            length += 1
        else:
            length += 0.5
    return length

def split_long_segments(elements, max_length=20):
    """
    处理元素列表，确保每两个换行符之间的长度不超过max_length。
    如果超过，则寻找最合适的空格替换为换行符。会直接replace!
    """
    current_length = 0.0
    space_positions = [] # 记录空格位置、该位置前的长度
    i = 0
    while i <= len(elements):
        if i == len(elements) or elements[i].get('type') == 0 and elements[i].get('orig') == '\n':
            if current_length > max_length and space_positions:
                # 寻找最能均匀分割行文本的空格
                n_cuts = current_length // max_length + 1
                n_cut_length = current_length / n_cuts
                sorted_spaces = sorted(space_positions, key=lambda x: (
                    0 if x[1] <= max_length else 1,
                    abs(x[1] - n_cut_length) if x[1] <= max_length else -x[1]
                ))
                best_position = sorted_spaces[0][0]
                elements[best_position]['orig'] = '\n'

                i = best_position # 从分割点之后开始
                current_length = 0.0
                space_positions = []
            else:
                current_length = 0.0
                space_positions = []
        
        elif i < len(elements):
            elem = elements[i]
            surface = elem.get('orig')
            elem_length = calculate_length(surface)
            if surface in (' ', '　') and elem.get('type') == 0:
                space_positions.append((i, current_length))
            current_length += elem_length
        i += 1

def non_silent_head_adjust(result_list, non_silent_ranges):
    """
    调整语音识别结果中的起始时间，确保乐句完全位于同一个非静音区间

    遍历 `result_list`，根据 `type` 字段将连续的非静音片段组合成乐句。
    对每个乐句，检查其时间范围是否被某个非静音区间完全覆盖，
    否则尝试将起始时间调整到能覆盖其结束位置的非静音区间的起始点。

    Args:
        result_list (List[Dict[str, Any]]): 语音识别结果片段列表。每个元素为字典，至少包含以下字段：
            - 'type' (int): 片段类型，0 表示静音分隔符，非 0 表示有效语音内容。
            - 'start' (str): 起始时间，格式为 "[MM:SS:CC]"。
            - 'end' (str): 结束时间，格式同 `start`。

        non_silent_ranges (List[Tuple[float, float]]): 非静音区间列表，
            每个区间为 `(开始秒数, 结束秒数)`，按开始时间升序排列。

    Returns:
        List[Dict[str, Any]]: 原地修改后的 `result_list`，
            其中部分元素的 'start' 字段可能被调整，其余字段保持不变。
    """
    if not non_silent_ranges:
        return result_list
    else:
        # 划分乐句
        i = si = 0
        sentences_list = []
        st = None
        while i < len(result_list):
            if result_list[i].get('type') == 0:
                if st:
                    sentences_list.append((si, i-1, st, result_list[i-1].get('end')))
                    st = None
            elif not st:
                si = i
                st = result_list[i].get('start')
            i += 1
        # 调整时间
        for inds, inde, sst, sen in sentences_list:
            sst = parse_time_to_hundredths(sst)
            sen = parse_time_to_hundredths(sen)
            interval_covered = False
            for ns_start, ns_end in non_silent_ranges:
                if int(ns_start * 100) > sst:
                    break
                # 检查非静音段是否覆盖整个区间
                if int(ns_start * 100) <= sst and int(np.ceil(ns_end * 100)) >= sen:
                    interval_covered = True
                    break
            if not interval_covered:
                end_covered = False
                for i in range(len(non_silent_ranges)):
                    ns_start = int(non_silent_ranges[i][0] * 100)
                    ns_end = int(np.ceil(non_silent_ranges[i][1] * 100))
                    if ns_start > sen:
                        break
                    if ns_start <= sen:
                        if ns_end >= sen:
                            end_covered = True
                            adjust_target = ns_start
                            break
                    elif ns_start <= parse_time_to_hundredths(result_list[inde].get('start')) <= ns_end:
                        end_covered = True
                        adjust_target = ns_start
                        break
                if not end_covered:
                    print('Errors ignored while trying to correct end sounds...')
                    break
                else:
                    adjust_target = min(parse_time_to_hundredths(result_list[inds]['end']), adjust_target)
                    result_list[inds]['start'] = format_hundredths_to_time_str(adjust_target)
        return result_list

def func_tail_correct_v250611(result_list, tail_mode=1):
    '旧版尾音调整预处理函数'
    if not result_list:
        return []
    if tail_mode == 1:
        for i in range(len(result_list)):
            if result_list[i]['type']==0:
                try:
                    if result_list[i-1].get('pron') and result_list[i-1]['type']!=0:
                        pre_vowel = result_list[i-1]['pron'][-1]
                        post_consonant = ''
                        if i < len(result_list)-1:
                            post_i = i + 1
                            while post_i < len(result_list):
                                if 'pron' in result_list[post_i] and len(result_list[post_i]['pron'])>=1:
                                    post_consonant = result_list[post_i]['pron'][0]
                                    break
                                else:
                                    post_i += 1
                        if pre_vowel!=post_consonant and post_consonant not in ('a', 'e', 'i', 'o', 'u'):
                            result_list[i]['pron'] = pre_vowel + 'h'
                except:
                    continue
    elif tail_mode == 2:
        for i in range(len(result_list)):
            if result_list[i]['type']==0:
                try: # 合理利用baseline尾音特性
                    if len(result_list[i-1]['pron'])>=1 and result_list[i-1]['type']!=0:
                        result_list[i]['pron'] = result_list[i-1]['pron'][-1] + 'h'
                except:
                    continue
    return result_list