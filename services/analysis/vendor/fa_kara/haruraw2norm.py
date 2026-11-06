from janome.tokenizer import Tokenizer
import nltk
from nltk.corpus import cmudict
import pykakasi
import pyphen
import pypinyin
from pypinyin import pinyin, Style
from uroman import Uroman
import re
import unicodedata
# import string

import indic_romanization
import label_rules
import script_romanization

kks = pykakasi.kakasi()
tokenizer = Tokenizer()
tail_pron = ""  # 'h'

phoneme_map = {
    "AA": "a",
    "AE": "a",
    "AH": "a",
    "AO": "o",
    "AW": "au",
    "AY": "ai",
    "B": "b",
    "CH": "ch",
    "D": "d",
    "DH": "z",
    "EH": "e",
    "ER": "a",
    "EY": "ei",
    "F": "f",
    "G": "g",
    "HH": "h",
    "IH": "i",
    "IY": "i",
    "JH": "j",
    "K": "k",
    "L": "r",
    "M": "m",
    "N": "n",
    "NG": "ng",
    "OW": "o",
    "OY": "oi",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "sh",
    "T": "t",
    "TH": "s",
    "UH": "u",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "y",
    "Z": "z",
    "ZH": "j",
}
try:
    cmu_dict = cmudict.dict()
except LookupError:
    nltk.download("cmudict")
    cmu_dict = cmudict.dict()
eng_dic = pyphen.Pyphen(lang="en_US")
uroman = Uroman()

newnums = [
    "①",
    "②",
    "③",
    "④",
    "⑤",
    "⑥",
    "⑦",
    "⑧",
    "⑨",
    "⑩",
    "⑪",
    "⑫",
    "⑬",
    "⑭",
    "⑮",
    "⑯",
    "⑰",
    "⑱",
    "⑲",
    "⑳",
    "㉑",
    "㉒",
    "㉓",
    "㉔",
    "㉕",
    "㉖",
    "㉗",
    "㉘",
    "㉙",
    "㉚",
]


def normalize_numbers(text):
    """将字符串中的各种数字字符转换为半角阿拉伯数字"""
    # 全角数字转半角
    fullwidth_to_half = str.maketrans("０１２３４５６７８９", "0123456789")
    text = text.translate(fullwidth_to_half)

    # 特殊数字转换映射表
    number_map = {
        # 分数
        "½": "0.5",
        "⅓": "0.333",
        "⅔": "0.666",
        "¼": "0.25",
        "¾": "0.75",
        "⅕": "0.2",
        "⅖": "0.4",
        "⅗": "0.6",
        "⅘": "0.8",
        "⅙": "0.166",
        "⅚": "0.833",
        "⅛": "0.125",
        "⅜": "0.375",
        "⅝": "0.625",
        "⅞": "0.875",
        # 罗马数字
        "Ⅰ": "1",
        "Ⅱ": "2",
        "Ⅲ": "3",
        "Ⅳ": "4",
        "Ⅴ": "5",
        "Ⅵ": "6",
        "Ⅶ": "7",
        "Ⅷ": "8",
        "Ⅸ": "9",
        "Ⅹ": "10",
        "Ⅺ": "11",
        "Ⅻ": "12",
        "Ⅼ": "50",
        "Ⅽ": "100",
        "Ⅾ": "500",
        "Ⅿ": "1000",
        # 汉字数字
        "零": "0",
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
        "十": "10",
        "百": "100",
        "千": "1000",
        "万": "10000",
        "亿": "100000000",
        # 上标/下标
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }

    # 构建转换表
    trans_table = str.maketrans(number_map)
    return text.translate(trans_table)


def number_to_english(number_str):
    """将数字字符串转换为英文单词"""
    try:
        if "." in number_str:
            num = float(number_str)
        else:
            num = int(number_str)
    except ValueError:
        print('Unable to process number "' + number_str + '"...')
        return tail_pron

    ones = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if isinstance(num, float):
        integer_part = int(num)
        decimal_part = round(num - integer_part, 3)
        integer_words = number_to_english(str(integer_part)) if integer_part > 0 else ""

        # 处理小数部分
        decimal_str = f"{decimal_part:.3f}"[2:]  # 获取小数点后三位
        decimal_words = " point"
        for digit in decimal_str:
            if digit == "0" and not decimal_words.endswith(" zero"):
                decimal_words += " zero"
            elif digit != "0":
                decimal_words += " " + ones[int(digit)]
        return (integer_words + decimal_words).strip()

    if num < 0:
        return "minus " + number_to_english(str(abs(num)))

    if num < 20:
        return ones[num]

    if num < 100:
        return tens[num // 10] + ((" " + ones[num % 10]) if num % 10 != 0 else "")

    if num < 1000:
        return (
            ones[num // 100]
            + " hundred"
            + (" and " + number_to_english(str(num % 100)) if num % 100 != 0 else "")
        )

    # 处理1000及以上
    scales = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (10**3, "thousand")]

    for scale_value, scale_name in scales:
        if num >= scale_value:
            return (
                number_to_english(str(num // scale_value))
                + " "
                + scale_name
                + (
                    " " + number_to_english(str(num % scale_value))
                    if num % scale_value != 0
                    else ""
                )
            )

    print('Unable to process number "' + number_str + '"...')
    return tail_pron


def is_english(text):
    return bool(re.match(r"^[a-zA-Z]+$", text))


def is_english_punctuation(char):
    return char == "'"  # in string.punctuation


def is_kanji(char):
    return (
        "\u4e00" <= char <= "\u9fff"
        or "\u3400" <= char <= "\u4dbf"
        or "\uf900" <= char <= "\ufaff"
        or char == "\u3005"
    )


def is_hiragana(char):
    return "\u3040" <= char <= "\u309f"


def is_katakana(char):
    return "\u30a0" <= char <= "\u30ff"


def is_kana(char):
    for i in char:
        if not is_hiragana(i) and not is_katakana(i) or i in ["・", "゠"]:  # , 'ー'
            return False
    return True


def is_number(char):
    # 目前支持判断单个字符
    if char in newnums:
        return False
    return char.isdigit()  # isnumeric


def get_norm_ruby(item):
    # 1:英文, 2:注音结构, 3:东亚文字（假名、中文汉字）, 4:数字, 5:隐式辅助注音
    if item["type"] == 2:
        return item["ruby"]
    if item["type"] in (3, 5):
        return item["orig"].lower() if is_english(item["orig"]) else item["orig"]
    if item["type"] == 1:
        return "".join(
            [char for char in item["orig"].strip() if not is_english_punctuation(char)]
        ).lower()
    return tail_pron


def get_norm_surface(item):
    if item["type"] in (1, 2, 3, 4, 5):
        return item["orig"]
    return ""


def min_error_split(target_list, s):
    n = len(s)
    m = len(target_list)

    # 初始化 DP 表
    # dp[i][k] 表示处理到字符串位置 i 时，已匹配 k 个目标项的最小错误数
    dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    # 记录回溯路径
    backtrack = [[None] * (m + 1) for _ in range(n + 1)]

    # 初始状态：空字符串匹配 0 个目标项
    dp[0][0] = 0

    # 动态规划填表
    for i in range(n + 1):
        for k in range(m + 1):
            if dp[i][k] == float("inf"):
                continue

            # 尝试匹配下一个目标项
            if k < m:
                target = target_list[k]
                # 处理空字符串目标项
                if target == "":
                    # 不消耗任何字符
                    if dp[i][k] < dp[i][k + 1]:
                        dp[i][k + 1] = dp[i][k]
                        backtrack[i][k + 1] = (i, k, "")
                else:
                    # 尝试所有可能的子串
                    for j in range(i + 1, n + 1):
                        segment = s[i:j]
                        # 计算错误成本（0~1 匹配~不匹配）
                        if segment == target:
                            cost = 0
                        elif target == tail_pron:
                            cost = min(len(segment) * 0.1, 1)
                        elif (
                            segment == "wa" and target == "ha" or segment == "e" and target == "ha"
                        ):
                            # 此处可添加当て字
                            cost = 0.1
                        else:
                            cost = 1
                        new_cost = dp[i][k] + cost
                        if new_cost < dp[j][k + 1]:
                            dp[j][k + 1] = new_cost
                            backtrack[j][k + 1] = (i, k, segment)

    # 回溯找到最佳分割
    if dp[n][m] == float("inf"):
        return None  # 无有效分割

    # 从终点回溯
    result = []
    i, k = n, m
    while k > 0:
        prev_i, prev_k, segment = backtrack[i][k]
        result.append(segment)
        i, k = prev_i, prev_k

    # 反转结果（因为是从后往前回溯）
    return result[::-1]


def sylla_split(kana_str, sokuon_split=False, hatsuon_split=True):
    kana_list = []
    i = 0
    n = len(kana_str)
    while i < n:
        current_char = kana_str[i]
        small_kana = [
            "ゃ",
            "ゅ",
            "ょ",
            "ぁ",
            "ぃ",
            "ぅ",
            "ぇ",
            "ぉ",
            "ー",
            "ャ",
            "ュ",
            "ョ",
            "ァ",
            "ィ",
            "ゥ",
            "ェ",
            "ォ",
        ]
        if not sokuon_split:
            small_kana += ["っ", "ッ"]
        if not hatsuon_split:
            small_kana += ["ん", "ン"]
        if current_char in small_kana:
            if i > 0:
                kana_list[-1] += current_char
            else:
                kana_list.append(current_char)
            i += 1
        else:
            kana_list.append(current_char)
            i += 1
    return kana_list


def convert_phoneme(ph):
    """去除音素中的重音标记并映射为罗马音"""
    base_ph = ph.rstrip("012")  # 移除数字重音标记
    return phoneme_map.get(base_ph, "")


def split_into_syllables_en(phonemes):
    """将英语音素序列拆分为音节"""
    vowels = [
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
    ]
    vowel_positions = []
    # 识别元音位置
    for i, ph in enumerate(phonemes):
        base_ph = ph.rstrip("012")
        if base_ph in vowels:
            vowel_positions.append(i)
    if not vowel_positions:
        return [phonemes]

    syllables = []
    prev_vowel_idx = -1

    # 按元音位置拆分音节
    for i, vowel_idx in enumerate(vowel_positions):
        if i == 0:
            # 首个音节
            onset = phonemes[:vowel_idx]
            vowel = [phonemes[vowel_idx]]
            syllables.append(onset + vowel)
            prev_vowel_idx = vowel_idx
        else:
            # 获取两个元音之间的辅音序列
            consonants = phonemes[prev_vowel_idx + 1 : vowel_idx]
            if consonants:
                onset_start = 0
                # 最大节首辅音原则
                if len(consonants) > 1:
                    # 将第一个辅音分配给前一个音节
                    syllables[-1].append(consonants[0])
                    onset_start = 1
                # 剩余辅音分配给后一个音节
                onset = consonants[onset_start:]
                vowel = [phonemes[vowel_idx]]
                syllables.append(onset + vowel)
            else:
                # 没有辅音，直接开始新音节
                syllables.append([phonemes[vowel_idx]])
            prev_vowel_idx = vowel_idx

    # 添加尾部剩余辅音到最后一个音节
    if prev_vowel_idx < len(phonemes) - 1:
        trailing = phonemes[prev_vowel_idx + 1 :]
        syllables[-1].extend(trailing)

    return syllables


def align_syllables_en(a, b):
    """简单对齐表面音节和发音音节"""
    if len(a) > len(b):
        long_list, short_list = a, b
        long_to_short = True
    elif len(b) > len(a):
        long_list, short_list = b, a
        long_to_short = False
    else:
        return list(zip(a, b))

    print("Ignored errors when dealing with the pronunciation of '" + "".join(a) + "'...")
    n_segments = len(short_list)
    total_elements = len(long_list)

    # 计算每段应包含的元素数
    base_size = total_elements // n_segments
    extra = total_elements % n_segments

    merged_list = []
    start = 0
    for i in range(n_segments):
        seg_size = base_size + (1 if i >= n_segments - extra else 0)  # 后 extra 段多一个元素
        segment = long_list[start : start + seg_size]
        merged = "".join(segment)
        merged_list.append(merged)
        start += seg_size

    if long_to_short:
        return list(zip(merged_list, short_list))
    else:
        return list(zip(short_list, merged_list))


def process_english_word(word, surf=True):
    if word == "a":
        return [("a", "a")]
    elif word == "A":
        return [("A", "ei")]

    hyphenated = eng_dic.inserted(word)
    surface_syllables = hyphenated.split("-")

    word_lower = word.lower()
    # if word_lower=='you':
    #     return [(word, 'iyu')]
    if word_lower not in cmu_dict:
        print("Word '" + word + "' not in the dictionary...")
        direct_syllables = [i.replace("'", "").lower() for i in surface_syllables]
        return list(zip(surface_syllables, direct_syllables))

    phonemes = cmu_dict[word_lower][0]
    syllables_phonemes = split_into_syllables_en(phonemes)
    syllables_romaji = []
    for syl in syllables_phonemes:
        romaji = "".join(convert_phoneme(p) for p in syl)  # p.rstrip('012').lower()
        syllables_romaji.append(romaji)

    if not surf:
        return "".join(syllables_romaji)

    # 对齐表面音节和发音音节
    return align_syllables_en(surface_syllables, syllables_romaji)


def hanzi_to_phonetic(text, heteronym=False):
    """One tone-less acoustic label per character of a Han run (script_romanization.mandarin_v1).

    Phrase-level readings come from pypinyin after OpenCC traditional→simplified conversion;
    the historical ``convert_pinyin_to_phonetic`` spelling (l→r, zhi→jru, ...) is retired.
    """
    return [
        label if is_kanji(char) else ""
        for char, label in zip(text, script_romanization.mandarin_readings(text))
    ]


def is_hangul(char):
    return "\uac00" <= char <= "\ud7a3" or "\u1100" <= char <= "\u11ff"


def is_arabic_script(char):
    return (
        "\u0600" <= char <= "\u06ff" or "\u0750" <= char <= "\u077f" or "\u08a0" <= char <= "\u08ff"
    )


def is_devanagari(char):
    return "\u0900" <= char <= "\u097f" or "\ua8e0" <= char <= "\ua8ff"


def is_gurmukhi(char):
    return "\u0a00" <= char <= "\u0a7f"


def _legacy_process_haruhi_line(line, lang="jaen", sokuon_split=False, hatsuon_split=True):

    def haruhi_eng_pron_func(result_list):
        # 英语分音节注音、数字注音
        new_list = []
        for item in result_list:
            if item.get("type") == 1:
                new_elements = get_norm_surface(item)
                new_list.extend(
                    [
                        {"orig": char, "type": 1, "pron": pron}
                        for char, pron in process_english_word(new_elements)
                    ]
                )
            elif item.get("type") == 4:
                en_nums = number_to_english(get_norm_surface(item)).split(" ")
                new_list.append(
                    {
                        "orig": get_norm_surface(item),
                        "type": 4,
                        "pron": "".join([process_english_word(i, surf=False) for i in en_nums]),
                    }
                )
            else:
                new_list.append(item)
        return new_list

    if lang == "auto":  # 后续考虑langdetect
        if any(is_hangul(char) for char in line):
            lang = "ko"
        elif any(is_arabic_script(char) for char in line):
            lang = "ur"
        elif any(is_devanagari(char) or is_gurmukhi(char) for char in line):
            lang = "indic"
        elif any(is_hiragana(char) or is_katakana(char) for char in line):
            lang = "jaen"
        else:
            lang = "zhen"

    result = []
    if lang in ["ja", "jaen"]:
        tokens = re.split(r"(\{.*?\}|\[.*?\])", line)

        for token in tokens:
            if not token:
                continue
            # 处理振假名结构 {漢字|假名}
            if token.startswith("{") and token.endswith("}"):
                content = token[1:-1]
                parts = content.split("|")
                assert len(parts) == 2, f"注音格式错误：{token}"
                kanji, ruby_text = parts
                ruby_text = sylla_split(ruby_text, sokuon_split, hatsuon_split)
                assert len(ruby_text) >= 1, "振假名为空"
                result.append({"orig": kanji, "type": 2, "ruby": ruby_text[0]})
                if len(ruby_text) >= 2:
                    for i in range(1, len(ruby_text)):
                        result.append({"orig": "", "type": 2, "ruby": ruby_text[i]})
            # 隐式注音 [字符|romaji]
            elif token.startswith("[") and token.endswith("]"):
                content = token[1:-1]
                parts = content.split("|")
                assert len(parts) == 2, f"注音格式错误：{token}"
                kanji, ruby_text = parts
                assert is_english(ruby_text), f"辅助注音格式错误：{token}"
                result.append({"orig": kanji, "type": 5, "pron": ruby_text})
            # 处理普通字符
            else:
                token = sylla_split(token, sokuon_split, hatsuon_split)
                if lang == "ja":
                    for char in token:
                        if is_kana(char) or is_kanji(char) or is_english(char):
                            result.append({"orig": char, "type": 3})
                        # elif is_number(char):
                        #     result.append({'orig': char, 'type': 4})
                        else:
                            result.append({"orig": char, "type": 0})
                elif lang == "jaen":
                    for char in token:
                        if is_kana(char) or is_kanji(char):
                            result.append({"orig": char, "type": 3})
                        elif is_english(char) or is_english_punctuation(char):
                            if result and result[-1].get("type") == 1:
                                result[-1]["orig"] += char
                            elif is_english(char):
                                result.append({"orig": char, "type": 1})
                            else:
                                result.append({"orig": char, "type": 0})
                        elif is_number(char):
                            if result and result[-1].get("type") == 4:
                                result[-1]["orig"] += char
                            else:
                                result.append({"orig": char, "type": 4})
                        else:
                            result.append({"orig": char, "type": 0})

        # 英语分音节注音、数字注音
        result = haruhi_eng_pron_func(result)

        # 标注单字罗马音
        postpron = None
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("type") in (0, 2, 3):
                ruby_now = get_norm_ruby(result[i])
                if result[i].get("type") != 0 and ruby_now and ruby_now[-1] in ("っ", "ッ"):
                    try:
                        pron = postpron[0]
                    except:
                        pron = "h"  # tail_pron
                    else:
                        if pron == "c":
                            pron = "t"
                    finally:
                        pron = kks.convert(ruby_now[:-1])[0]["hepburn"] + pron
                else:
                    pron = kks.convert(ruby_now)[0]["hepburn"]
                result[i]["pron"] = pron.replace("-", "")  # 长音的错误token
            postpron = result[i]["pron"]

        # 通用读音修正（は，へ）
        line_pron_list = [item["pron"] for item in result]
        line_surface = "".join([get_norm_surface(i) for i in result])
        line_roma = "".join(
            [
                i["hepburn"]
                for i in kks.convert(
                    "".join([token.phonetic for token in tokenizer.tokenize(line_surface)])
                )
            ]
        )
        line_roma_proc = min_error_split(line_pron_list, line_roma)
        for i in range(len(result)):
            if result[i]["type"] == 3:
                try:
                    if result[i]["orig"] == "は" and line_roma_proc[i] == "wa":
                        result[i]["pron"] = "wa"
                    elif result[i]["orig"] == "へ" and line_roma_proc[i] == "e":
                        result[i]["pron"] = "e"
                except:
                    print("Ignored errors when trying to correct ha and he...")
    elif lang == "ko":
        for char in line:
            if is_hangul(char):
                pronunciation = uroman.romanize_string(char).replace("-", "").replace(" ", "")
                result.append({"orig": char, "type": 5, "pron": pronunciation})
            elif is_english(char) or is_english_punctuation(char):
                if result and result[-1].get("type") == 1:
                    result[-1]["orig"] += char
                elif is_english(char):
                    result.append({"orig": char, "type": 1})
                else:
                    result.append({"orig": char, "type": 0})
            elif is_number(char):
                if result and result[-1].get("type") == 4:
                    result[-1]["orig"] += char
                else:
                    result.append({"orig": char, "type": 4})
            else:
                result.append({"orig": char, "type": 0})
        result = haruhi_eng_pron_func(result)

    elif lang in ("ur", "hi", "pa", "indic"):
        if lang == "ur":
            script_test = is_arabic_script
        elif lang == "hi":
            script_test = is_devanagari
        elif lang == "pa":
            script_test = is_gurmukhi
        else:
            script_test = lambda char: is_devanagari(char) or is_gurmukhi(char)
        for char in line:
            if script_test(char) and unicodedata.category(char)[0] in {"L", "M"}:
                if result and result[-1].get("type") == 6:
                    result[-1]["orig"] += char
                else:
                    result.append({"orig": char, "type": 6})
            elif is_english(char) or is_english_punctuation(char):
                if result and result[-1].get("type") == 1:
                    result[-1]["orig"] += char
                elif is_english(char):
                    result.append({"orig": char, "type": 1})
                else:
                    result.append({"orig": char, "type": 0})
            elif is_number(char):
                if result and result[-1].get("type") == 4:
                    result[-1]["orig"] += char
                else:
                    result.append({"orig": char, "type": 4})
            else:
                result.append({"orig": char, "type": 0})
        for item in result:
            if item.get("type") == 6:
                item["pron"] = (
                    uroman.romanize_string(item["orig"]).replace("-", "").replace(" ", "")
                )
        result = haruhi_eng_pron_func(result)

    elif lang == "zhen":
        tokens = re.split(r"(\[.*?\])", line)

        for token in tokens:
            if not token:
                continue
            if token.startswith("[") and token.endswith("]"):
                content = token[1:-1]
                parts = content.split("|")
                assert len(parts) == 2, f"注音格式错误：{token}"
                kanji, ruby_text = parts
                assert is_english(ruby_text), f"辅助注音格式错误：{token}"
                result.append({"orig": kanji, "type": 5, "pron": ruby_text})
            else:
                for char in token:
                    if is_kanji(char):
                        result.append({"orig": char, "type": 3, "pron": hanzi_to_phonetic(char)[0]})
                    elif is_english(char) or is_english_punctuation(char):
                        if result and result[-1].get("type") == 1:
                            result[-1]["orig"] += char
                        elif is_english(char):
                            result.append({"orig": char, "type": 1})
                        else:
                            result.append({"orig": char, "type": 0})
                    elif is_number(char):
                        if result and result[-1].get("type") == 4:
                            result[-1]["orig"] += char
                        else:
                            result.append({"orig": char, "type": 4})
                    else:
                        result.append({"orig": char, "type": 0})

        # 数字先给英语吧
        result = haruhi_eng_pron_func(result)

    return result


def _japanese_words(text, sokuon_split=False, hatsuon_split=True, rules=None):
    """Janome owns word boundaries/readings; never apportion kanji readings.

    Label rules v2: a token-final sokuon (待っ|てる, split by Janome) geminates the next token's
    first consonant instead of reading ``tsu`` (matteru, not matsu te ru; the legacy per-character
    path carried it the same way), and the particle を reads ``o``.
    """
    rules = label_rules.resolve(rules)
    tokens = []
    for token in tokenizer.tokenize(text):
        surface = token.surface
        reading = token.reading
        if token.part_of_speech.startswith("助詞"):
            reading = {"は": "ワ", "へ": "エ", "を": "オ" if rules.v2 else "ヲ"}.get(
                surface, reading
            )
        if reading == "*":
            reading = surface
        pron = "".join(part["hepburn"] for part in kks.convert(reading)).replace("-", "")
        tokens.append([surface, reading, pron])
    if rules.v2:
        for index, (surface, reading, pron) in enumerate(tokens):
            if (
                surface.endswith(("っ", "ッ"))
                and pron.endswith("tsu")
                and reading.endswith(("っ", "ッ", "ッ"))
            ):
                following = next((p for _, _, p in tokens[index + 1 :] if p), "")
                consonant = following[:1] if following[:1] in "bcdfghjkmnprstwyz" else ""
                if consonant == "c":
                    consonant = "t"
                tokens[index][2] = pron[:-3] + consonant
    result = []
    for surface, reading, pron in tokens:
        mora = sylla_split(reading, sokuon_split, hatsuon_split)
        # Prefix romanization supplies boundaries only when it is an exact
        # prefix of the contextual label. Gemination/long vowels stay grouped
        # when splitting would change the trained character stream.
        cuts = [0]
        for i in range(1, len(mora)):
            prefix = "".join(p["hepburn"] for p in kks.convert("".join(mora[:i]))).replace("-", "")
            if pron.startswith(prefix) and len(prefix) > cuts[-1]:
                cuts.append(len(prefix))
        cuts.append(len(pron))
        labels = [pron[a:b] for a, b in zip(cuts, cuts[1:]) if b > a]
        if not labels:
            result.append({"orig": surface, "type": 3, "pron": pron})
            continue
        # Kana suffixes are literal anchors, not guessed isolated kanji readings.
        suffix = re.search(r"[ぁ-ゖァ-ヺー]+$", surface)
        display = [surface] + [""] * (len(labels) - 1)
        if suffix:
            kana = sylla_split(suffix.group(), sokuon_split, hatsuon_split)
            kana_labels = [
                "".join(p["hepburn"] for p in kks.convert(k)).replace("-", "") for k in kana
            ]
            if kana_labels and labels[-len(kana) :] == kana_labels:
                n = len(labels) - len(kana)
                if n > 0:
                    display = [surface[: suffix.start()]] + [""] * (n - 1) + kana
                elif suffix.start() == 0:
                    display = kana
        result.extend(
            {"orig": surf, "type": 3, "pron": label} for surf, label in zip(display, labels)
        )
    return result


def _latin_context(runs, index):
    """(previous, next) Latin word of a run in its line, lower-cased, '#' at the line edges (v3)."""

    def word(step):
        position = index + step
        while 0 <= position < len(runs):
            if runs[position][0] == "latin":
                return runs[position][1].lower().replace("’", "'").replace("ʼ", "'")
            position += step
        return "#"

    return word(-1), word(1)


def process_haruhi_line(line, lang="jaen", sokuon_split=False, hatsuon_split=True, rules=None):
    """Script-local normalization using the existing acoustic spelling alphabet.

    Auto Han-only runs default to Mandarin (not language detection) unless the
    line contains kana anywhere, Cantonese-only characters, or the hint says
    ja/jaen (Japanese) or yue (Jyutping); explicit zh keeps Han runs Mandarin.
    Foreign scripts cannot steal a line. Auto Latin is spelling/transliteration,
    not assumed English; only explicit en/jaen/zhen enables CMU for ASCII words.
    Unknown ASCII words use spelling boundaries without changing the
    concatenated acoustic labels. Thai, Arabic-script, Cyrillic and the other
    letter scripts are handled by script_romanization (see its docstring).

    ``rules`` selects the label rules (``label_rules.RULES``): ``None``/``"v1"`` is today's output byte
    for byte; ``"v2"`` adds pronunciation labels for en/de/fr hints, number/acronym/vocable/symbol
    reading, Indic and Tamil pronunciation transliteration, Arabic ʿayn/hamza vowels, the Japanese
    sokuon carry, Cantonese jyu, ñ → ny; ``"v2-orthographic"`` is v2 with spelled Latin words.
    ``"v3"`` / ``"v3-orthographic"`` add the v3 fixes (label_rules.py docstring), including the
    per-line language of en/de/fr lines.
    """
    lang = (lang or "auto").lower()
    rules = label_rules.resolve(rules)
    if rules.v3:
        # v3: a line of an en/de/fr song in another language uses that language's rules (fastText).
        lang = label_rules.line_language(line, lang)
    result = []
    # Line-level context: kana anywhere makes Han runs Japanese (explicit zh/yue/zhen keep
    # them Chinese); Cantonese-only characters or the yue hint select Jyutping; Turkish letters
    # select uroman's Turkish rules for Latin runs. Deterministic, not language identification.
    line_kana = lang not in ("zh", "yue", "zhen") and any(is_kana(c) for c in line)
    cantonese = lang == "yue" or any(c in script_romanization.CANTONESE_MARKERS for c in line)
    latin_code = script_romanization.latin_lcode(line, lang)

    def script(char):
        if is_kanji(char) or is_kana(char):
            return "cjk"
        if is_hangul(char):
            return "ko"
        if is_arabic_script(char):
            return "ur"
        if is_devanagari(char) or is_gurmukhi(char):
            return "indic"
        if "LATIN" in unicodedata.name(char, "") or char == "'" or (rules.v2 and char in "’ʼ"):
            return "latin"
        if unicodedata.category(char).startswith("M"):
            return "mark"
        if is_number(char):
            return "number"
        if unicodedata.category(char).startswith("L"):
            name = script_romanization.script_name(char)
            if name == "THAI":
                return "thai"
            if name in script_romanization.UROMAN_SCRIPTS:
                return "uroman:" + name
        return "other"

    # Annotations are opaque boundaries: never reinterpret an explicit reading.
    for part in re.split(r"(\{[^{}]*\}|\[[^\[\]]*\])", line):
        if not part:
            continue
        if part.startswith(("{", "[")) and part.endswith(("}", "]")):
            result.extend(_legacy_process_haruhi_line(part, "jaen", sokuon_split, hatsuon_split))
            continue
        runs = []
        for char in part:
            kind = script(char)
            if kind == "mark" and runs:
                kind = runs[-1][0]
            if runs and runs[-1][0] == kind:
                runs[-1][1] += char
            else:
                runs.append([kind, char])
        if rules.v2:
            runs = label_rules.merge_number_runs(runs)
        # v3: per-word language overrides (English words in non-Latin-script songs, foreign word runs in
        # en/de/fr lines); empty for every other rule set.
        word_languages = label_rules.latin_run_languages(runs, lang, rules) if rules.v3 else {}
        skip_next = False
        for run_index, (kind, text) in enumerate(runs):
            if skip_next:
                skip_next = False
                continue
            following = runs[run_index + 1] if run_index + 1 < len(runs) else None
            if kind == "cjk":
                japanese = lang in ("ja", "jaen") or line_kana or any(is_kana(c) for c in text)
                if japanese:
                    result.extend(_japanese_words(text, sokuon_split, hatsuon_split, rules))
                else:
                    readings = (
                        script_romanization.cantonese_readings(text, v2=rules.v2)
                        if cantonese
                        else hanzi_to_phonetic(text)
                    )
                    # Do not silently truncate if the library cannot map a run; a character
                    # without a reading stays unlabelled so the timed adapter masks it.
                    if len(readings) != len(text):
                        result.append({"orig": text, "type": 0})
                    else:
                        result.extend(
                            {"orig": c, "type": 3, "pron": p} if p else {"orig": c, "type": 0}
                            for c, p in zip(text, readings)
                        )
            elif kind == "ur":
                items = _legacy_process_haruhi_line(text, kind, sokuon_split, hatsuon_split)
                for item in items:
                    if item.get("type") == 6:
                        item["pron"] = script_romanization.arabic_label(
                            item["orig"], uroman, v2=rules.v2
                        )
                result.extend(items)
            elif kind == "number" and rules.v2:
                # v2: words of the hint's language, read like words (years, ordinals, "$5", "10,000").
                ordinal = (
                    following is not None
                    and following[0] == "latin"
                    and following[1].lower() in label_rules.ORDINAL_SUFFIXES.get(lang, set())
                )
                currency = (
                    result[-1]["orig"]
                    if result
                    and result[-1].get("orig") in label_rules.CURRENCY
                    and result[-1].get("type") == 0
                    else None
                )
                result.extend(
                    label_rules.number_items(text, lang, rules, ordinal=ordinal, currency=currency)
                )
                if ordinal:
                    result.append({"orig": following[1], "type": 0})
                    skip_next = True
            elif kind == "indic" and rules.v2:
                items = _legacy_process_haruhi_line(text, kind, sokuon_split, hatsuon_split)
                for item in items:
                    if item.get("type") == 6:
                        label = (
                            indic_romanization.gurmukhi_label(item["orig"])
                            if is_gurmukhi(item["orig"][0])
                            else indic_romanization.devanagari_label(item["orig"])
                        )
                        if label:
                            item["pron"] = label
                        else:
                            item.pop("pron", None)
                            item["type"] = 0
                result.extend(items)
            elif kind in ("ko", "indic", "number"):
                result.extend(
                    _legacy_process_haruhi_line(
                        text, "zhen" if kind == "number" else kind, sokuon_split, hatsuon_split
                    )
                )
            elif kind == "thai":
                items = script_romanization.thai_items(text, uroman)
                # Apple times a standalone repetition mark (" ๆ") as its own word: it repeats
                # the previous syllable's sound.
                if items and items[0]["orig"] == "ๆ" and not items[0].get("pron"):
                    previous = next((i["pron"] for i in reversed(result) if i.get("pron")), "")
                    if previous:
                        items[0]["pron"] = previous
                result.extend(items)
            elif kind.startswith("uroman:"):
                name = kind.split(":", 1)[1]
                if rules.v2 and name == "TAMIL":
                    label = indic_romanization.tamil_label(text)
                    result.append(
                        {"orig": text, "type": 8, "pron": label}
                        if label
                        else {"orig": text, "type": 0}
                    )
                    continue
                lcode = script_romanization.UROMAN_SCRIPTS[name]
                result.append(
                    {
                        "orig": text,
                        "type": 8,
                        "pron": script_romanization.uroman_label(text, uroman, lcode),
                    }
                )
            elif kind == "latin":
                if (
                    text.isascii()
                    and lang in ("jaen", "zhen")
                    or (text.isascii() and lang == "en" and not rules.v2)
                ):
                    result.extend(
                        _legacy_process_haruhi_line(text, "zhen", sokuon_split, hatsuon_split)
                    )
                else:

                    def spell(text=text):
                        # Uroman folds accents without deleting their base letters.
                        # Keep orig byte-for-byte for conservative timed-span mapping.
                        source = text
                        if rules.v2:
                            source = script_romanization.latin_v2_source(text, latin_code)
                        romanized = (
                            uroman.romanize_string(source, lcode=latin_code)
                            if latin_code
                            else uroman.romanize_string(source)
                        )
                        pron = romanized.lower().replace("'", "").replace("-", "").replace(" ", "")
                        if rules.v2:
                            pron = pron.replace("’", "").replace("ʼ", "")
                        parts = eng_dic.inserted(text).split("-") if text.isascii() else [text]
                        labels = [p.lower().replace("'", "") for p in parts]
                        if "".join(labels) == pron and all(labels):
                            return [
                                {"orig": p, "type": 5, "pron": label}
                                for p, label in zip(parts, labels)
                            ]
                        return [{"orig": text, "type": 5, "pron": pron}]

                    if rules.v3:
                        result.extend(
                            label_rules.latin_items(
                                text,
                                word_languages.get(run_index, lang),
                                rules,
                                spell,
                                context=_latin_context(runs, run_index),
                            )
                        )
                    elif rules.v2:
                        result.extend(label_rules.latin_items(text, lang, rules, spell))
                    else:
                        result.extend(spell())
            elif (
                rules.v2
                and text in label_rules.SYMBOL_WORDS
                and lang not in ("ja", "jaen", "zhen", "ko", "yue", "zh")
            ):
                if (
                    text in label_rules.CURRENCY
                    and following is not None
                    and following[0] == "number"
                ):
                    result.append(
                        {"orig": text, "type": 0}
                    )  # read with the amount: "$5" → five dollars
                else:
                    result.extend(label_rules.symbol_items(text, lang, rules))
            else:
                result.append({"orig": text, "type": 0})
    # Ruby sokuon may depend on the following run/annotation. Resolve after
    # segmentation, using the same consonant convention as the legacy path.
    following = ""
    for item in reversed(result):
        ruby = item.get("ruby", "")
        if item["type"] == 2 and ruby.endswith(("っ", "ッ")):
            consonant = following[:1] or "h"
            if consonant == "c":
                consonant = "t"
            item["pron"] = (
                "".join(p["hepburn"] for p in kks.convert(ruby[:-1])).replace("-", "") + consonant
            )
        following = item.get("pron", "")
    return result


if __name__ == "__main__":
    input_string = "{阻|はば}むも[の|n]は{無|な}い {身|み}{勝|かっ}{手|て}に More love, more jump!"
    parsed = process_haruhi_line(input_string)
    print(parsed)
