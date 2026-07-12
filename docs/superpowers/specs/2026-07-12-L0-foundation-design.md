# L0 · 地基（Foundation）设计文档

日期：2026-07-12
状态：待用户审阅

---

## 1. 目标

一个 Web app：系统问一道面试题 → 用户对着摄像头回答 → 系统输出统计量（转录 + 凝视 + 语速）→ **不存任何音视频**。

导师目标：跑通这个 = "a great foundation"，12 周内可完成。原则：**一步一验证**，每步能独立确认"跑通了"再进下一步。

## 2. 三条硬约束（贯穿全程，不可违反）

1. **必须是 Web app** —— 能部署到公网、被试点链接就能用、不装任何东西。
2. **绝不落库原始音视频** —— 实时处理完即丢，只存统计量。（免走伦理审批的红线）
3. **两个消融开关（RAG 开/关、多模态开/关）从一开始就埋进去** —— L1/L2 跑消融要靠它们。

## 3. 架构总览

**浏览器负责"采集 + 凝视"，服务器负责"转文字 + 汇总 + 落库"。** 原始视频永远不离开浏览器；音频只在"结束"时发一次给服务器转录，转完即丢。第 2 条红线在架构层面天然满足。

```
┌─────────── 浏览器 (纯 HTML/JS) ───────────┐        ┌────────── 服务器 (FastAPI) ──────────┐
│  getUserMedia 取摄像头+麦克风              │        │                                      │
│  MediaPipe FaceLandmarker → 逐帧凝视      │══WS══▶ │  会话控制 / 出题                     │
│  MediaRecorder 录音(内存)                 │  凝视   │  凝视样本聚合                        │
│                                           │  样本   │  faster-whisper 转录(带时间戳)       │
│  点"结束" → 发送音频 blob ───────────────│══HTTP▶ │  算 WPM → 合并凝视 → 存 JSON         │
│  显示统计量                    ◀──────────│         │  (音频转完即丢, 不写磁盘)            │
└───────────────────────────────────────────┘        └──────────────────────────────────────┘
```

**一处刻意偏离清单**：清单设想"服务器端 MediaPipe"，本设计改为**浏览器端 MediaPipe**。好处：红线更硬（视频不出终端）、服务器更轻。凝视本就是代理，不影响研究价值。

## 4. 组件划分（职责单一、可独立验证）

| 组件 | 职责 | 依赖 |
|------|------|------|
| `frontend/` 静态页 | 取流、显示自检画面、跑 MediaPipe、录音、WS 通信 | 浏览器 API、MediaPipe(WASM, CDN) |
| `server/app.py` | FastAPI 入口 + WebSocket + HTTP 路由 | FastAPI, uvicorn |
| `server/asr.py` | 音频块迭代器 → 带时间戳转录 | faster-whisper |
| `server/metrics.py` | 从转录算 WPM、从凝视样本算占比，做窗口平滑 | 纯 Python |
| `server/session.py` | 出题、汇总统计量、写 JSON | 标准库 |
| `server/config.py` | **两个消融开关** + 题库 + 模型配置 | 标准库 |

## 5. 一轮完整数据流（Step 5→6）

1. 点"开始" → 服务器从题库选一题 → 前端用浏览器 `speechSynthesis` TTS 念出来（先纯文字显示也行，TTS 后加）。
2. 用户回答期间：
   - 浏览器每帧算"是否看镜头"，把**紧凑的凝视样本**（`{t, looking:bool, yaw?, pitch?}`）经 WebSocket 持续发给服务器。
   - MediaRecorder 在**内存**里录音（不写文件）。
3. 点"结束"（手动，L0 不做端点检测）→ 音频 blob 经 HTTP 发给服务器。
4. 服务器：
   - `asr.py` 转录（词级时间戳）
   - `metrics.py` 算 WPM + 合并本轮凝视占比（窗口平滑）
   - `session.py` 存成一份 `results/session_<id>.json`
   - 音频数据转完即丢，不落盘。
5. 返回统计量给前端显示 → **人工 `ls` 确认磁盘上没有任何音视频文件**。

> **说明（哪些是必要的、哪些是提前铺路）**：L0 的凝视统计是**结束后才汇总**的（实时 HUD 是 L2 的事），功能上等价于"结束时把凝视样本和音频一并上传"。之所以 L0 就让凝视走 WebSocket 持续发送，是**为 L2 的实时 HUD 预留链路**（且 Step 1 本就要搭 WS）。心里要清楚：此时 WS 传凝视是提前铺路，不是 L0 的功能刚需。

## 6. ASR 接口设计（关键：为流式预留）

**L0 用批处理**（"攒齐音频再转"），但**接口按流式设计**，避免 L2 才发现整条链路要重写。

```python
# server/asr.py
from dataclasses import dataclass
from typing import Iterable, Iterator

@dataclass
class AudioChunk:
    pcm: bytes          # 一小段音频（PCM/编码后）
    t_start: float      # 该块在本轮回答里的起始秒
    sample_rate: int

@dataclass
class Word:
    text: str
    t_start: float      # 词级时间戳（算 WPM 用）
    t_end: float

@dataclass
class Transcript:
    text: str
    words: list[Word]

def transcribe(chunks: Iterable[AudioChunk]) -> Transcript:
    """
    L0 实现：把 chunks 攒齐 → 一次性喂给 faster-whisper → 返回带词级时间戳的转录。
    调用方看到的已经是流式接口（接收音频块迭代器）。
    L1/L2：改成边收 chunk 边转录、增量 yield 结果，仅改本函数内部，不动链路。
    """
    ...
```

**设计约定**：L0 = 批处理；**L1/L2 必须换成流式（边说边转）以支撑评估协议的实时延迟 P50/P99**。届时只改 `asr.py` 内部实现，调用方接口不变。

## 7. 消融开关（Step 埋点，L1/L2 才真用）

`config.py` 两个布尔，从一开始贯穿接口，输出的统计量里都带上当前 flag：

```python
RAG_ENABLED = False        # L0 恒 False，占位；L1 才接 RAG
MULTIMODAL_ENABLED = True   # 见下方语义
```

**`MULTIMODAL_ENABLED` 语义（已修正）**：语速也是非语言信号（paralinguistic），关多模态时应一起关。

- `True` → 统计量含 **凝视 + 语速**
- `False` → **只有转录（纯文本）**

对应研究问题 **RQ2b（多模态 vs 纯文本反馈）**。

**`False` 时的具体行为（L0 实现）**：前端**照常跑** MediaPipe / 录音，**在服务器端过滤掉**凝视与语速字段（实现最简单，链路不变）。

**⚠ 别混为一谈（架构清单里区分过的两个开关）**：
- `MULTIMODAL_ENABLED` 是**消融开关**——决定统计量/反馈里含不含非语言信号。
- 到 L2 做用户研究时，"关多模态"条件下前端**不应显示 HUD**——但那是**HUD 显示开关**，和消融开关是**两个不同的东西**。L0 先只实现消融开关；HUD 显示开关留到 L2。

## 8. Step 4 感知细节

- **凝视代理**：MediaPipe FaceLandmarker 拿人脸关键点 → 推出头部朝向/视线朝向 → 判定"是否在看镜头"。**做到"看镜头 vs 看别处能明显区分"即可**，不追求精确视线追踪。
- **判定阈值（显式定义，待调）**：头部/视线朝向偏离镜头中心的角度 **< `GAZE_ON_CAMERA_DEG` 度** 即判定为"看镜头"。
  - `GAZE_ON_CAMERA_DEG` 是**待定参数**，Step 4 要多调几次确定，放进 `config.py`。
  - **论文方法章节需交代**此阈值的取值与调法（凝视是代理，判定规则必须可复现）。
- **语速（WPM）**：从 Step 3 的词级时间戳算，`words 数 / 时长(分钟)`。
- **两路信号都做窗口平滑**（逐帧/逐词数据很抖）。
- **验证**：看镜头 vs 看别处 → 凝视指标明显不同；说快 vs 说慢 → WPM 明显不同。

## 9. 目录结构

```
L0-foundation/
├── server/
│   ├── app.py         # FastAPI + WebSocket + HTTP
│   ├── asr.py         # 流式接口 + L0 批处理实现
│   ├── metrics.py     # WPM / 凝视聚合 / 平滑
│   ├── session.py     # 出题 + 汇总 + 落库
│   └── config.py      # 消融开关 + 题库 + 模型配置
├── frontend/
│   ├── index.html
│   ├── app.js         # getUserMedia / MediaPipe / MediaRecorder / WS
│   └── style.css      # 极简，功能够用即冻结
├── results/           # 只存统计量 JSON，进 .gitignore（见 §10）
├── requirements.txt
├── .gitignore
└── README.md          # 跑起来 + 逐步验证 + 数据安全说明
```

## 10. 数据安全与合规（README 必须写明）

- **`results/` 是被试的研究数据**，绝不能进 git 仓库 → 加入 `.gitignore`，README 写明此原因。
- README 交代：
  - 统计量存在哪（`results/session_<id>.json`），字段含义。
  - **只存统计量**（转录文本 + 凝视占比 + 平均 WPM + flags），**不存任何音视频**。
  - "数据安全存储"如何保证：存储位置、访问控制、后续可加密/匿名化的位置 —— 对齐伦理清单里"数据安全存储"这条。
- **验证红线（= L0 完成标志之一）**：走完一轮后，磁盘上找不到任何音视频文件，此条必须人工确认。

## 11. 部署（Step 7）

推荐 **cloudflared 隧道**：从 M 系列 Mac 起一条公网隧道，别人点链接即用、不装东西，whisper 跑在本机（性能够），零云成本。正式云部署留到 L1 之后。

验证：发链接给同学 → 他打开 → 授权摄像头 → 答一题 → 你拿到他的统计量 → 他没装任何东西。

## 12. 落地节奏与逐步验证点（严格按清单 Step 1→7）

| Step | 做什么 | 跑通标志（独立验证） |
|------|--------|----------------------|
| 1 | 骨架：前端 + FastAPI + WebSocket | 网页发 "hello" → 后端回 "hi" |
| 2 | 取摄像头 + 麦克风 | 弹权限 → 同意后看到自己画面、麦有信号（不写文件） |
| 3 | ASR（批处理，流式接口） | 说一句话 → 屏幕出对应文字（带时间戳） |
| 4 | 凝视(浏览器 MediaPipe) + 语速 | 看镜头 vs 别处 / 说快 vs 说慢 → 指标明显不同 |
| 5 | 出题 + 会话流程(手动结束 + TTS) | 点开始→问一题→回答→点结束 |
| 6 | 汇总统计量 + 落库 | 拿到一份统计量 JSON；磁盘无任何音视频 ← 必须人工确认 |
| 7 | 部署上线 | 别人电脑打开链接答一题 → 你拿到他的统计量 |

## 13. 实战提醒（清单原则）

- 别在前端好看上花时间：功能够用就冻结。
- 别追求完美：凝视是代理、结束用按钮、题目手写几道。
- 早点部署：网络/浏览器权限/跨设备兼容的坑越早踩越好。
- 两个消融开关现在就留位。

## 14. 技术栈确认

- ASR：**本地 faster-whisper**（零 API 费用、音频不出本机、契合"不落库"）
- 凝视：**浏览器内 MediaPipe FaceLandmarker**（视频不出终端）
- 前端：**纯 HTML/JS**（免构建）
- 后端：**FastAPI + uvicorn**
- 环境：**Python 3.11 或 3.12**（faster-whisper 依赖 CTranslate2，对最新 Python 支持通常滞后；3.13 可能装不上，先写死更稳妥。本机默认是 3.13，需另建 3.11/3.12 虚拟环境）；Homebrew 可用；需安装 ffmpeg（faster-whisper 解码用）。
- **不需要 Node**：前端纯 HTML/JS、免构建，L0 无前端构建链。

## 15. L0 之后

L0 = 可远程访问、伦理合规、能产出行为统计量的 Web 系统。之后进 L1：加 LLM + RAG，生成纯文本反馈，搭裁判 + 专家校准 + RAG 消融。
