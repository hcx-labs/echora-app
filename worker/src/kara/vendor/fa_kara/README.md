# FA-Kara
一个基于注音歌词文本和人声音频的自动打轴工具，主要参考了[yohane](https://github.com/Japan7/yohane)、[Forced-Alignment-For-NicoKara](https://github.com/oHEILIo/Forced-Alignment-For-NicoKara/)。

建议使用此工具处理日语歌曲，但底层模型实际上不限语种。

## 使用说明
配置好Python环境后，准备以下两个文件（可参考示例）：
#### 1. 音频
- **格式及命名**: `i.wav`
- **要求**: 从歌曲中分离出的人声，例如可以用[UVR](https://ultimatevocalremover.com/)或MSST生成

#### 2. 歌词
- **格式及命名**: `i.txt`
- **要求**: 中文、英文无需处理；日文需要有振假名注音，可使用[SUG](https://github.com/karaoke-studio/StrangeUtaGame)或RhythmicaLyrics出力->春日向けテキスト；其他文本可以注罗马音，如`[3σ|srisigma]`（这部分工作甚至可以用大模型完成）；一行歌词不能同时出现中文和日文；文本尽可能与音频内容保持一致

将音频、歌词和各个`.py`文件放在同一目录，运行指令：
``` shell
python main.py
```

若运行成功，该目录下会生成三个文件：
- `o.ass`: 可在Aegisub中灵活编辑
- `o_rlf.lrc`: 可在RhythmicaLyrics中灵活编辑
- `o_ruby.lrc`: 可直接在NicoKaraMaker3使用，默认提前150ms

### 高级选项
运行指令时还可以添加参数。例如基座模型效果不理想，要使用微调模型，你可以使用如下指令：
``` shell
python main.py -hf 'D:\你的本地模型路径'
```
方便起见，建议从[Hugging Face官网](https://huggingface.co/NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn)或[镜像站](https://hf-mirror.com/NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn/tree/main)手动下载模型到本地。

更多选项请用`-h`查看。

## 环境配置
参考`requirements.txt`，请根据实际情况调整。

如果你从未接触过Python，可按照如下步骤配置环境：
1. 安装[Python](https://www.python.org/)（推荐版本3.13），并配置好环境变量；
2. 根据操作系统与GPU情况，安装对应的[PyTorch](https://pytorch.org/get-started/locally/)；
3. 再安装其他库，如运行指令`pip install janome librosa nltk pykakasi pyphen pypinyin transformers`；
4. 第一次运行程序时会自动下载模型与字典，这可能需要更多的时间。

## 模型简介
主要使用librosa进行音频预处理，结合Janome、pykakasi、NLTK、Pyphen、PyPinyin标注歌词文本读音，输入基于PyTorch的[MMS_FA](https://arxiv.org/abs/2305.13516)进行推理。

目前的效果尚不尽如人意，欢迎交流。

## 项目结构
```text
FA-Kara/
├── align.py                 # 核心对齐引擎（基座 MMS_FA）
├── align_yohane.py          # 核心对齐引擎（yohane 微调模型）
├── ass2lrc.py               # 已弃用，ass转RL特定格式
├── haruraw2norm.py          # 歌词注音与解析，并转成标准结构
├── lrcfmt.py                # 非春日向け格式歌词处理
├── main.py                  # 主程序入口，解析参数并执行对齐流程
├── norm2ass.py              # 已打轴的标准结构转ass
├── norm2lrc.py              # 已打轴的标准结构转lrc
├── utils_audio.py           # 音频辅助函数（librosa 工具）
├── utils_basic.py           # 基础辅助函数（时间格式转换等简单文本处理）
├── requirements.txt         # Python 依赖列表参考
├── i.txt                    # 示例：输入歌词文本文件
└── i.wav                    # 示例：输入人声音频文件
```
