# L0 地基（Foundation）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 一个 Web app —— 系统问一道面试题 → 用户对着摄像头回答 → 输出统计量（转录 + 凝视 + 语速）→ 不存任何音视频 → 能公网部署给别人用。

**Architecture:** 浏览器（纯 HTML/JS）负责采集 + 浏览器内 MediaPipe 算凝视；服务器（FastAPI）负责本地 faster-whisper 转录 + 汇总统计量 + 落库 JSON。原始视频永不离开浏览器，音频转完即丢。WebSocket 传凝视样本（为 L2 实时 HUD 预留），音频在"结束"时 HTTP 上传。

**Tech Stack:** Python 3.11/3.12、FastAPI、uvicorn、faster-whisper、MediaPipe FaceLandmarker (浏览器 WASM)、纯 HTML/JS、pytest、cloudflared（部署）。

**参考设计文档：** `docs/superpowers/specs/2026-07-12-L0-foundation-design.md`

---

## 文件结构

```
L0-foundation/
├── server/
│   ├── __init__.py
│   ├── app.py         # FastAPI + WebSocket + HTTP 路由 + 静态文件
│   ├── config.py      # 消融开关 + 题库 + 模型/阈值配置
│   ├── asr.py         # 流式接口(AudioChunk 迭代器) + L0 批处理实现
│   ├── metrics.py     # WPM / 凝视聚合 / 平滑（纯函数，TDD）
│   └── session.py     # 出题 + 汇总统计量 + 落库 + no-media 断言
├── frontend/
│   ├── index.html
│   ├── app.js         # getUserMedia / MediaPipe / MediaRecorder / WS
│   └── style.css
├── tests/
│   ├── test_metrics.py
│   ├── test_session.py
│   └── test_asr_interface.py
├── results/           # 只存统计量 JSON（.gitignore 已排除）
├── requirements.txt
├── pytest.ini
├── run.sh             # 本地启动
├── tunnel.sh          # cloudflared 公网隧道
└── README.md          # 跑起来 + 逐步验证 + 数据安全说明
```

**测试策略：** 纯 Python 逻辑（`metrics.py`、`session.py`、`asr.py` 接口形状）用 pytest 做 TDD。浏览器/硬件相关步骤（摄像头、MediaPipe、录音、WS 端到端、部署）用清单的"跑通标志"做**手动验证**——这些无法可靠单测。每个手动验证点都在计划里写清"怎么算跑通"。

---

## Task 0: 项目环境与骨架目录

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `server/__init__.py`, `.python-version`

- [ ] **Step 1: 确认可用的 Python 3.11/3.12 解释器**

Run:
```bash
cd "/Users/kong/Desktop/Glasgow毕业论文/L0-foundation"
brew list python@3.12 >/dev/null 2>&1 || brew install python@3.12
$(brew --prefix python@3.12)/bin/python3.12 --version
```
Expected: `Python 3.12.x`

- [ ] **Step 2: 建虚拟环境（用 3.12，避开 3.13 的 CTranslate2 兼容坑）**

Run:
```bash
$(brew --prefix python@3.12)/bin/python3.12 -m venv .venv
.venv/bin/python --version
```
Expected: `Python 3.12.x`

- [ ] **Step 3: 装 ffmpeg（faster-whisper/PyAV 解码音频用）**

Run:
```bash
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
ffmpeg -version | head -1
```
Expected: 打印 ffmpeg 版本

- [ ] **Step 4: 写 requirements.txt**

Create `requirements.txt`:
```
fastapi==0.115.6
uvicorn[standard]==0.34.0
faster-whisper==1.1.1
python-multipart==0.0.20
pytest==8.3.4
httpx==0.28.1
```

- [ ] **Step 5: 安装依赖**

Run:
```bash
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```
Expected: 全部成功，无 CTranslate2 编译错误

- [ ] **Step 6: 建包目录与 pytest 配置**

Create `server/__init__.py`（空文件）。

Create `pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v
```

Create `.python-version`:
```
3.12
```

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pytest.ini server/__init__.py .python-version
git commit -m "chore: 项目环境与依赖 (Python 3.12, faster-whisper, FastAPI)"
```

---

## Task 1 (清单 Step 1): 骨架 —— FastAPI + WebSocket + 静态前端，能互发消息

**Files:**
- Create: `server/app.py`, `frontend/index.html`, `frontend/app.js`, `frontend/style.css`
- Test: `tests/test_ws_echo.py`

- [ ] **Step 1: 写 WebSocket echo 的失败测试**

Create `tests/test_ws_echo.py`:
```python
from fastapi.testclient import TestClient
from server.app import app


def test_ws_echo_replies_hi_to_hello():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "text": "hello"})
        reply = ws.receive_json()
    assert reply["type"] == "echo"
    assert reply["text"] == "hi"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_ws_echo.py -v`
Expected: FAIL（`ModuleNotFoundError: server.app` 或 app 无 /ws）

- [ ] **Step 3: 写最小 app.py 让测试通过**

Create `server/app.py`:
```python
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="L0 Foundation")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "hello":
                await ws.send_json({"type": "echo", "text": "hi"})
            else:
                await ws.send_json({"type": "echo", "text": msg.get("text", "")})
    except WebSocketDisconnect:
        return


# 静态前端挂在最后，避免盖过 API 路由
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_ws_echo.py -v`
Expected: PASS

- [ ] **Step 5: 写最小前端**

Create `frontend/index.html`:
```html
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>L0 面试地基</title>
  <link rel="stylesheet" href="/style.css" />
</head>
<body>
  <h1>L0 面试地基</h1>

  <section id="conn">
    <button id="btn-hello">发 hello 测试链路</button>
    <span id="ws-status">未连接</span>
    <pre id="log"></pre>
  </section>

  <section id="capture" hidden>
    <video id="selfview" autoplay muted playsinline></video>
    <div>麦克风电平：<meter id="mic" min="0" max="1" value="0"></meter></div>
  </section>

  <script src="/app.js"></script>
</body>
</html>
```

Create `frontend/style.css`:
```css
body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
video { width: 320px; border: 1px solid #ccc; border-radius: 8px; transform: scaleX(-1); }
pre { background: #f4f4f4; padding: .5rem; border-radius: 6px; min-height: 2rem; }
button { padding: .5rem 1rem; }
```

Create `frontend/app.js`:
```javascript
const log = (m) => {
  document.getElementById("log").textContent += m + "\n";
};

const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onopen = () => { document.getElementById("ws-status").textContent = "已连接"; };
ws.onclose = () => { document.getElementById("ws-status").textContent = "已断开"; };
ws.onmessage = (e) => { log("收到: " + e.data); };

document.getElementById("btn-hello").onclick = () => {
  ws.send(JSON.stringify({ type: "hello", text: "hello" }));
  log("发送: hello");
};
```

- [ ] **Step 6: 手动验证（清单 Step 1 跑通标志）**

Run: `.venv/bin/uvicorn server.app:app --reload --port 8000`
打开 `http://localhost:8000` → 点"发 hello 测试链路"。
Expected: 页面 log 显示 `发送: hello` 和 `收到: {"type":"echo","text":"hi"}`，状态显示"已连接"。

- [ ] **Step 7: Commit**

```bash
git add server/app.py frontend/ tests/test_ws_echo.py
git commit -m "feat(step1): 骨架 FastAPI+WebSocket+静态前端，hello/hi 互通"
```

---

## Task 2 (清单 Step 2): 取摄像头 + 麦克风（只取流，不写文件）

**Files:**
- Modify: `frontend/app.js`, `frontend/index.html`

- [ ] **Step 1: 前端请求媒体权限并显示自检画面**

在 `frontend/app.js` 末尾追加：
```javascript
// ---- Step 2: 取摄像头 + 麦克风（只取流，绝不写文件/落库）----
let mediaStream = null;

async function initMedia() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: true,
  });
  document.getElementById("capture").hidden = false;
  document.getElementById("selfview").srcObject = mediaStream;

  // 麦克风电平自检
  const audioCtx = new AudioContext();
  const src = audioCtx.createMediaStreamSource(mediaStream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  src.connect(analyser);
  const buf = new Uint8Array(analyser.fftSize);
  const meter = document.getElementById("mic");
  (function tick() {
    analyser.getByteTimeDomainData(buf);
    let peak = 0;
    for (const v of buf) peak = Math.max(peak, Math.abs(v - 128) / 128);
    meter.value = peak;
    requestAnimationFrame(tick);
  })();
}

document.getElementById("btn-media").onclick = () =>
  initMedia().catch((e) => log("媒体权限失败: " + e));
```

在 `frontend/index.html` 的 `#conn` section 里加一个按钮：
```html
    <button id="btn-media">开启摄像头 + 麦克风</button>
```

- [ ] **Step 2: 手动验证（清单 Step 2 跑通标志）**

刷新 `http://localhost:8000` → 点"开启摄像头 + 麦克风"。
Expected: 浏览器弹权限请求 → 同意后看到自己的画面；对着麦说话，电平条 `#mic` 有跳动。

- [ ] **Step 3: 红线自检**

确认 `app.js` 里**没有任何** `MediaRecorder` 保存、没有 `download`、没有把流上传落盘的代码——此步只取流。
Run（应无匹配）：`grep -n "download\|saveAs\|localStorage" frontend/app.js || echo "OK: 无落盘代码"`
Expected: `OK: 无落盘代码`

- [ ] **Step 4: Commit**

```bash
git add frontend/app.js frontend/index.html
git commit -m "feat(step2): 取摄像头+麦克风自检画面与电平，只取流不落盘"
```

---

## Task 3 (清单 Step 3): ASR —— 流式接口 + L0 批处理实现

**Files:**
- Create: `server/config.py`, `server/asr.py`
- Modify: `server/app.py`, `frontend/app.js`, `frontend/index.html`
- Test: `tests/test_asr_interface.py`

- [ ] **Step 1: 写 config.py（消融开关 + 模型 + 阈值 + 题库 + 目录）**

Create `server/config.py`:
```python
"""L0 全局配置。两个消融开关从一开始就在这里，贯穿全程。"""
from pathlib import Path

# ---- 两个消融开关（L1/L2 跑消融要靠它们）----
RAG_ENABLED = False          # L0 恒 False，占位；L1 才接 RAG
MULTIMODAL_ENABLED = True    # True=统计量含凝视+语速；False=只有转录(纯文本)。对应 RQ2b

# ---- 凝视代理阈值（Step 4 要调，论文方法章节需交代）----
GAZE_ON_CAMERA_DEG = 15.0    # 头部/视线偏离镜头中心 < 此角度 → 判定"看镜头"

# ---- ASR（本地 faster-whisper）----
WHISPER_MODEL = "small.en"   # 英文面试；CPU 上可换 "base.en" 提速
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_DEVICE = "cpu"       # Mac CPU；有 GPU 可改

# ---- 题库（先手写几道，不建题库）----
QUESTIONS = [
    "Tell me about a challenging project you worked on and your role in it.",
    "Describe a time you had a conflict with a teammate and how you handled it.",
    "Why are you interested in this position?",
    "What is a technical skill you recently learned, and how did you learn it?",
]

# ---- 落库目录（只存统计量，绝不存音视频）----
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
```

- [ ] **Step 2: 写 asr 接口的失败测试（验证接口形状：接收音频块迭代器）**

Create `tests/test_asr_interface.py`:
```python
from server.asr import AudioChunk, Transcript, Word, transcribe


def test_transcribe_accepts_chunk_iterator_and_returns_transcript(monkeypatch):
    """接口按流式设计：入参是 AudioChunk 迭代器。此测试用假模型，只验接口形状与聚合。"""
    captured = {}

    class FakeSegment:
        def __init__(self):
            self.words = [
                _FakeWord("hello", 0.0, 0.5),
                _FakeWord("world", 0.5, 1.0),
            ]

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            captured["called"] = True
            return [FakeSegment()], None

    monkeypatch.setattr("server.asr._get_model", lambda: FakeModel())

    chunks = iter([
        AudioChunk(data=b"\x00\x01", t_start=0.0),
        AudioChunk(data=b"\x02\x03", t_start=0.5),
    ])
    result = transcribe(chunks)

    assert captured["called"] is True
    assert isinstance(result, Transcript)
    assert result.text == "hello world"
    assert result.words == [
        Word("hello", 0.0, 0.5),
        Word("world", 0.5, 1.0),
    ]


class _FakeWord:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end
```

- [ ] **Step 3: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_asr_interface.py -v`
Expected: FAIL（`server.asr` 不存在）

- [ ] **Step 4: 写 asr.py（流式接口 + L0 批处理实现）**

Create `server/asr.py`:
```python
"""ASR：L0 用批处理，但接口按流式设计（接收 AudioChunk 迭代器）。
L1/L2 换流式时只改本文件内部（边收 chunk 边增量转录），调用方接口不变。
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable

from server import config

_model = None  # 懒加载，避免测试/导入时下载模型


@dataclass
class AudioChunk:
    data: bytes                 # L0: 整段编码音频(webm/opus)；L1/L2 流式时为小段
    t_start: float = 0.0        # 该块在本轮回答里的起始秒（为流式对齐预留）
    sample_rate: int | None = None


@dataclass
class Word:
    text: str
    t_start: float              # 词级时间戳（算 WPM 用）
    t_end: float


@dataclass
class Transcript:
    text: str
    words: list[Word]


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _model


def transcribe(chunks: Iterable[AudioChunk]) -> Transcript:
    """L0 实现：攒齐所有 chunk → 一次性喂给 faster-whisper → 带词级时间戳的转录。"""
    audio_bytes = b"".join(c.data for c in chunks)
    model = _get_model()
    segments, _info = model.transcribe(
        io.BytesIO(audio_bytes),
        word_timestamps=True,
        language="en",
    )
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            words.append(Word(text=w.word.strip(), t_start=w.start, t_end=w.end))
    text = " ".join(w.text for w in words)
    return Transcript(text=text, words=words)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_asr_interface.py -v`
Expected: PASS

- [ ] **Step 6: 在 app.py 加 /transcribe 端点（音频 blob → 单块迭代器 → 转录）**

在 `server/app.py` 顶部 import 区加：
```python
from fastapi import UploadFile
from server.asr import AudioChunk, transcribe
```

在 `app.mount(...)` **之前**加：
```python
@app.post("/transcribe")
async def transcribe_endpoint(audio: UploadFile):
    raw = await audio.read()
    # L0：整段音频包成单个 chunk 的迭代器（接口已是流式形状）
    result = transcribe(iter([AudioChunk(data=raw, t_start=0.0)]))
    return {"text": result.text, "words": [w.__dict__ for w in result.words]}
```

- [ ] **Step 7: 前端录音并上传（说完一段 → 出文字）**

在 `frontend/app.js` 末尾追加：
```javascript
// ---- Step 3: 录音(内存) → 结束时上传转录 ----
let recorder = null;
let audioChunks = [];

function startRecording() {
  audioChunks = [];
  recorder = new MediaRecorder(mediaStream, { mimeType: "audio/webm" });
  recorder.ondataavailable = (e) => { if (e.data.size) audioChunks.push(e.data); };
  recorder.start();
  log("开始录音");
}

async function stopRecordingAndTranscribe() {
  const done = new Promise((res) => (recorder.onstop = res));
  recorder.stop();
  await done;
  const blob = new Blob(audioChunks, { type: "audio/webm" });
  audioChunks = []; // 立即丢弃内存音频
  const form = new FormData();
  form.append("audio", blob, "answer.webm");
  const resp = await fetch("/transcribe", { method: "POST", body: form });
  const data = await resp.json();
  document.getElementById("transcript").textContent = data.text || "(无转录)";
  log("转录完成");
}

document.getElementById("btn-rec-start").onclick = startRecording;
document.getElementById("btn-rec-stop").onclick = () =>
  stopRecordingAndTranscribe().catch((e) => log("转录失败: " + e));
```

在 `frontend/index.html` 的 `#capture` section 里加：
```html
    <div>
      <button id="btn-rec-start">开始录音</button>
      <button id="btn-rec-stop">结束并转录</button>
    </div>
    <p>转录：<span id="transcript"></span></p>
```

- [ ] **Step 8: 手动验证（清单 Step 3 跑通标志）**

重启 uvicorn（首次会下载 whisper 模型，稍等）→ 开启摄像头麦克风 → 点"开始录音" → 对着麦说一句英文 → 点"结束并转录"。
Expected: `#transcript` 出现大致准确的对应文字；`data.words` 带时间戳（Step 5 算 WPM 要用）。

- [ ] **Step 9: Commit**

```bash
git add server/config.py server/asr.py server/app.py frontend/ tests/test_asr_interface.py
git commit -m "feat(step3): ASR 流式接口+faster-whisper 批处理实现，录音→转录跑通"
```

---

## Task 4 (清单 Step 4): 非语言感知 —— 凝视(浏览器 MediaPipe) + 语速(WPM)

**Files:**
- Create: `server/metrics.py`
- Modify: `server/app.py`, `frontend/app.js`, `frontend/index.html`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: 写 metrics 的失败测试（WPM / 凝视占比 / 平滑，纯函数）**

Create `tests/test_metrics.py`:
```python
from server.asr import Word
from server.metrics import GazeSample, compute_wpm, gaze_on_camera_ratio, smooth_looking


def test_compute_wpm_basic():
    # 4 个词跨 0.0~2.0 秒 = 2 秒 = 1/30 分钟 → 120 WPM
    words = [
        Word("a", 0.0, 0.4), Word("b", 0.5, 0.9),
        Word("c", 1.0, 1.4), Word("d", 1.5, 2.0),
    ]
    assert round(compute_wpm(words)) == 120


def test_compute_wpm_empty_is_zero():
    assert compute_wpm([]) == 0.0


def test_gaze_on_camera_ratio():
    samples = [
        GazeSample(0.0, True), GazeSample(0.1, True),
        GazeSample(0.2, False), GazeSample(0.3, True),
    ]
    assert gaze_on_camera_ratio(samples) == 0.75


def test_gaze_on_camera_ratio_empty_is_zero():
    assert gaze_on_camera_ratio([]) == 0.0


def test_smooth_looking_removes_single_frame_jitter():
    # 中间一帧 False 是抖动，多数投票(窗口3)应抹平成 True
    samples = [
        GazeSample(0.0, True), GazeSample(0.1, True),
        GazeSample(0.2, False), GazeSample(0.3, True), GazeSample(0.4, True),
    ]
    smoothed = smooth_looking(samples, window=3)
    assert [s.looking for s in smoothed] == [True, True, True, True, True]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL（`server.metrics` 不存在）

- [ ] **Step 3: 写 metrics.py**

Create `server/metrics.py`:
```python
"""统计量纯函数：WPM、凝视占比、凝视平滑。逐帧/逐词数据很抖，故做窗口平滑。"""
from __future__ import annotations

from dataclasses import dataclass

from server.asr import Word


@dataclass
class GazeSample:
    t: float            # 秒
    looking: bool       # 是否在看镜头（浏览器端已按 GAZE_ON_CAMERA_DEG 判定）


def compute_wpm(words: list[Word]) -> float:
    """从词级时间戳算语速（每分钟词数）。"""
    if len(words) < 2:
        return 0.0
    duration_sec = words[-1].t_end - words[0].t_start
    if duration_sec <= 0:
        return 0.0
    return len(words) / (duration_sec / 60.0)


def smooth_looking(samples: list[GazeSample], window: int = 5) -> list[GazeSample]:
    """滑动窗口多数投票，抹平单帧抖动。窗口取奇数。"""
    if window < 2 or len(samples) < window:
        return list(samples)
    half = window // 2
    out: list[GazeSample] = []
    for i, s in enumerate(samples):
        lo = max(0, i - half)
        hi = min(len(samples), i + half + 1)
        votes = [x.looking for x in samples[lo:hi]]
        majority = sum(votes) * 2 >= len(votes)
        out.append(GazeSample(t=s.t, looking=majority))
    return out


def gaze_on_camera_ratio(samples: list[GazeSample]) -> float:
    """看镜头时间占比。"""
    if not samples:
        return 0.0
    return sum(1 for s in samples if s.looking) / len(samples)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: PASS（5 个测试全绿）

- [ ] **Step 5: 前端加载 MediaPipe FaceLandmarker 并逐帧算凝视**

在 `frontend/app.js` 末尾追加（MediaPipe 从 CDN 以 ES module 动态载入）：
```javascript
// ---- Step 4: 浏览器内 MediaPipe 算凝视代理，样本经 WS 发出（为 L2 HUD 预留链路）----
const GAZE_ON_CAMERA_DEG = 15.0; // 与 server/config.py 保持一致，Step 4 要调
let faceLandmarker = null;
let gazeTimer = null;
let sessionT0 = null;

async function initGaze() {
  const vision = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/vision_bundle.mjs"
  );
  const { FaceLandmarker, FilesetResolver } = vision;
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    },
    runningMode: "VIDEO",
    outputFacialTransformationMatrixes: true,
    numFaces: 1,
  });
  log("MediaPipe 就绪");
}

// 从 4x4 变换矩阵估计头部 yaw/pitch（度），作为"看镜头"代理
function headAnglesDeg(matrix) {
  const m = matrix; // column-major 长度16
  const yaw = Math.atan2(m[8], m[10]) * 180 / Math.PI;
  const pitch = Math.atan2(-m[9], Math.sqrt(m[8] ** 2 + m[10] ** 2)) * 180 / Math.PI;
  return { yaw, pitch };
}

function startGazeLoop() {
  const video = document.getElementById("selfview");
  sessionT0 = performance.now();
  gazeTimer = setInterval(() => {
    if (!faceLandmarker || video.readyState < 2) return;
    const res = faceLandmarker.detectForVideo(video, performance.now());
    const t = (performance.now() - sessionT0) / 1000;
    let looking = false;
    const mats = res.facialTransformationMatrixes;
    if (mats && mats.length) {
      const { yaw, pitch } = headAnglesDeg(mats[0].data);
      looking = Math.abs(yaw) < GAZE_ON_CAMERA_DEG && Math.abs(pitch) < GAZE_ON_CAMERA_DEG;
      document.getElementById("gaze-live").textContent =
        `${looking ? "看镜头" : "看别处"} (yaw=${yaw.toFixed(0)}, pitch=${pitch.toFixed(0)})`;
    }
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "gaze", t, looking }));
    }
  }, 100); // 10 fps 足够
}

function stopGazeLoop() {
  if (gazeTimer) clearInterval(gazeTimer);
  gazeTimer = null;
}

document.getElementById("btn-gaze").onclick = () =>
  initGaze().then(startGazeLoop).catch((e) => log("MediaPipe 失败: " + e));
```

在 `frontend/index.html` 的 `#capture` section 里加：
```html
    <div><button id="btn-gaze">开始凝视检测</button> <span id="gaze-live"></span></div>
```

- [ ] **Step 6: 后端 WS 收集凝视样本（本轮内累积）**

修改 `server/app.py` 的 `ws_endpoint`，把 `type=="gaze"` 的样本累积到连接局部列表；同时导入 GazeSample：

顶部 import 加：
```python
from server.metrics import GazeSample
```

把 `ws_endpoint` 替换为：
```python
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    gaze_samples: list[GazeSample] = []
    ws.state.gaze_samples = gaze_samples  # 供同连接的其他处理读取（Step 6 汇总用）
    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            if mtype == "hello":
                await ws.send_json({"type": "echo", "text": "hi"})
            elif mtype == "gaze":
                gaze_samples.append(GazeSample(t=msg["t"], looking=bool(msg["looking"])))
            else:
                await ws.send_json({"type": "echo", "text": msg.get("text", "")})
    except WebSocketDisconnect:
        return
```

> 注：`ws.state` 在 Starlette WebSocket 上可用（`ws.state` 是 `State()` 实例）。若版本不支持，改用模块级 `dict[ws_id -> samples]`。Step 6 会把汇总逻辑并进来。

- [ ] **Step 7: 手动验证（清单 Step 4 跑通标志）**

刷新页面 → 开启摄像头 → 点"开始凝视检测"（首次下载 MediaPipe 模型，稍等）。
Expected:
- 看镜头 vs 看别处 → `#gaze-live` 在"看镜头/看别处"间明显切换，yaw/pitch 数值明显不同。
- 说快 vs 说慢：录两段（各说 5~10 个词，一快一慢）分别转录，用浏览器 console 手算 `words.length / 时长` 或等 Step 6 汇总，WPM 明显不同。

> ⚠ 这是 L0 最 fiddly 的一步。若"看别处仍判成看镜头"，回 `config.py` / `app.js` 调 `GAZE_ON_CAMERA_DEG`（如 10~20 之间试）。做到"能区分"即可。

- [ ] **Step 8: Commit**

```bash
git add server/metrics.py server/app.py frontend/ tests/test_metrics.py
git commit -m "feat(step4): 浏览器MediaPipe凝视代理+WPM，metrics纯函数TDD，WS收样本"
```

---

## Task 5 (清单 Step 5): 出题 + 会话流程（手动结束 + TTS）

**Files:**
- Create: `server/session.py`
- Modify: `server/app.py`, `frontend/app.js`, `frontend/index.html`
- Test: `tests/test_session.py`（本 Task 只测出题部分）

- [ ] **Step 1: 写出题的失败测试**

Create `tests/test_session.py`:
```python
from server import config
from server.session import pick_question


def test_pick_question_returns_one_from_bank():
    q = pick_question()
    assert q in config.QUESTIONS


def test_pick_question_deterministic_with_index():
    assert pick_question(index=0) == config.QUESTIONS[0]
    assert pick_question(index=1) == config.QUESTIONS[1]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_session.py -v`
Expected: FAIL（`server.session` 不存在）

- [ ] **Step 3: 写 session.py 的出题部分**

Create `server/session.py`:
```python
"""会话：出题 + 汇总统计量 + 落库。Step 5 先做出题，Step 6 补汇总/落库。"""
from __future__ import annotations

import random

from server import config


def pick_question(index: int | None = None) -> str:
    """从手写题库选一题。index 给定则确定性返回（测试/复现用）。"""
    if index is not None:
        return config.QUESTIONS[index]
    return random.choice(config.QUESTIONS)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_session.py -v`
Expected: PASS

- [ ] **Step 5: 后端 WS 支持"开始"→ 出题**

在 `server/app.py` 顶部 import 加：
```python
from server.session import pick_question
```

在 `ws_endpoint` 的消息分支里，`elif mtype == "gaze":` 之前加：
```python
            elif mtype == "start":
                question = pick_question()
                ws.state.question = question
                await ws.send_json({"type": "question", "text": question})
```

- [ ] **Step 6: 前端会话流程（开始 → 出题 + TTS → 结束）**

在 `frontend/app.js` 末尾追加：
```javascript
// ---- Step 5: 会话流程 出题→听→结束（手动结束，不做端点检测）----
function speak(text) {
  try {
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US";
    speechSynthesis.speak(u);
  } catch (e) { log("TTS 不可用: " + e); }
}

// 收到题目时朗读并显示
const origOnMessage = ws.onmessage;
ws.onmessage = (e) => {
  origOnMessage(e);
  let msg;
  try { msg = JSON.parse(e.data); } catch { return; }
  if (msg.type === "question") {
    document.getElementById("question").textContent = msg.text;
    speak(msg.text);
  }
};

document.getElementById("btn-start").onclick = () => {
  document.getElementById("transcript").textContent = "";
  ws.send(JSON.stringify({ type: "start" }));
  startRecording();
  if (faceLandmarker) startGazeLoop();
  log("会话开始");
};

document.getElementById("btn-end").onclick = async () => {
  stopGazeLoop();
  await stopRecordingAndTranscribe();
  log("会话结束");
};
```

在 `frontend/index.html` 的 `#capture` section 里加：
```html
    <hr />
    <p>题目：<strong id="question"></strong></p>
    <button id="btn-start">开始（出题并回答）</button>
    <button id="btn-end">结束</button>
```

- [ ] **Step 7: 手动验证（清单 Step 5 跑通标志）**

刷新 → 开启摄像头 + 凝视检测 → 点"开始" → 听到/看到题目 → 回答 → 点"结束"。
Expected: 点开始后系统问一道题（TTS 念出 + 文字显示），回答后点结束能走完一轮（转录出现）。

- [ ] **Step 8: Commit**

```bash
git add server/session.py server/app.py frontend/ tests/test_session.py
git commit -m "feat(step5): 出题+会话流程(开始/结束手动)+浏览器TTS"
```

---

## Task 6 (清单 Step 6): 汇总统计量 + 落库 + no-media 断言（= L0 完成）

**Files:**
- Modify: `server/session.py`, `server/app.py`, `frontend/app.js`, `frontend/index.html`
- Test: `tests/test_session.py`（追加汇总/落库/消融/no-media 测试）

- [ ] **Step 1: 追加汇总 + 落库 + 消融 + no-media 的失败测试**

在 `tests/test_session.py` 末尾追加：
```python
import json

from server.asr import Transcript, Word
from server.metrics import GazeSample
from server.session import assert_no_media, build_summary, save_summary


def _sample_inputs():
    transcript = Transcript(
        text="hello world",
        words=[Word("hello", 0.0, 0.5), Word("world", 0.5, 1.0)],
    )
    gaze = [GazeSample(0.0, True), GazeSample(0.1, False)]
    return transcript, gaze


def test_build_summary_multimodal_on_has_gaze_and_wpm():
    transcript, gaze = _sample_inputs()
    s = build_summary("sess1", "Q?", transcript, gaze, multimodal=True, rag=False)
    assert s["transcript"] == "hello world"
    assert "gaze_on_camera_ratio" in s
    assert "avg_wpm" in s
    assert s["flags"] == {"multimodal": True, "rag": False}


def test_build_summary_multimodal_off_is_text_only():
    transcript, gaze = _sample_inputs()
    s = build_summary("sess1", "Q?", transcript, gaze, multimodal=False, rag=False)
    assert s["transcript"] == "hello world"
    assert "gaze_on_camera_ratio" not in s   # 语速也是非语言信号，一起关
    assert "avg_wpm" not in s
    assert s["flags"] == {"multimodal": False, "rag": False}


def test_save_summary_writes_json_and_no_media(tmp_path):
    transcript, gaze = _sample_inputs()
    s = build_summary("sess1", "Q?", transcript, gaze, multimodal=True, rag=False)
    path = save_summary(s, results_dir=tmp_path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["session_id"] == "sess1"
    # 落库目录里绝不能有任何音视频文件
    assert_no_media(tmp_path)  # 不抛异常即通过


def test_assert_no_media_raises_when_media_present(tmp_path):
    (tmp_path / "leak.webm").write_bytes(b"x")
    try:
        assert_no_media(tmp_path)
        assert False, "应当检测到音视频文件并抛错"
    except AssertionError as e:
        assert "media" in str(e).lower() or "音视频" in str(e)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_session.py -v`
Expected: FAIL（`build_summary` / `save_summary` / `assert_no_media` 未定义）

- [ ] **Step 3: 在 session.py 补汇总 / 落库 / no-media 断言**

在 `server/session.py` 追加（顶部 import 补充）：
```python
import json
from datetime import datetime, timezone
from pathlib import Path

from server.asr import Transcript
from server.metrics import GazeSample, compute_wpm, gaze_on_camera_ratio, smooth_looking

_MEDIA_EXTS = {".wav", ".mp3", ".webm", ".mp4", ".mov", ".m4a", ".ogg", ".avi"}


def build_summary(
    session_id: str,
    question: str,
    transcript: Transcript,
    gaze_samples: list[GazeSample],
    *,
    multimodal: bool,
    rag: bool,
) -> dict:
    """汇总一题的统计量。multimodal=False → 只留转录（凝视+语速都关）。"""
    summary = {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "transcript": transcript.text,
        "flags": {"multimodal": multimodal, "rag": rag},
    }
    if multimodal:
        smoothed = smooth_looking(gaze_samples)
        summary["gaze_on_camera_ratio"] = round(gaze_on_camera_ratio(smoothed), 3)
        summary["avg_wpm"] = round(compute_wpm(transcript.words), 1)
    return summary


def save_summary(summary: dict, results_dir: Path | None = None) -> Path:
    """把统计量写成 JSON。只存统计量，绝不存音视频。"""
    results_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"session_{summary['session_id']}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return path


def assert_no_media(results_dir: Path | None = None) -> None:
    """红线自检：落库目录里绝不能出现任何音视频文件。"""
    results_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
    if not results_dir.exists():
        return
    offenders = [p for p in results_dir.rglob("*") if p.suffix.lower() in _MEDIA_EXTS]
    assert not offenders, f"检测到音视频文件(media)，违反红线: {offenders}"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_session.py -v`
Expected: PASS（全部）

- [ ] **Step 5: 后端把一轮汇总落库（结束时触发）**

L0 里音频转录在 HTTP `/transcribe`，凝视样本在 WS。最简单的串联：让 `/transcribe` 兼作"结束汇总"——前端结束时把 `session_id` 和本轮凝视样本一起 POST 过来，服务器汇总落库。

把 `server/app.py` 的 `/transcribe` 端点替换为：
```python
import json as _json

from server import config
from server.session import assert_no_media, build_summary, save_summary
from server.metrics import GazeSample


@app.post("/transcribe")
async def transcribe_endpoint(
    audio: UploadFile,
    session_id: str = Form(...),
    question: str = Form(""),
    gaze: str = Form("[]"),
):
    raw = await audio.read()
    result = transcribe(iter([AudioChunk(data=raw, t_start=0.0)]))
    del raw  # 音频转完即丢，不落盘

    gaze_samples = [GazeSample(t=g["t"], looking=bool(g["looking"]))
                    for g in _json.loads(gaze)]
    summary = build_summary(
        session_id, question, result, gaze_samples,
        multimodal=config.MULTIMODAL_ENABLED, rag=config.RAG_ENABLED,
    )
    path = save_summary(summary)
    assert_no_media()  # 落库后立即自检红线
    return {"summary": summary, "saved_to": str(path.name)}
```

顶部 import 补 `Form`：
```python
from fastapi import Form, UploadFile
```

> 前端改为在结束时把凝视样本随音频一起 POST（凝视 WS 通道仍保留，为 L2 HUD 预留；见设计文档 §5）。

- [ ] **Step 6: 前端结束时携带 session_id + 凝视样本上传，并显示统计量**

修改 `frontend/app.js`：在文件顶部加会话状态与本地凝视缓存：
```javascript
// 本轮 session 状态（Step 6）
let sessionId = null;
let gazeBuffer = [];
```

在 `startGazeLoop` 里 `ws.send(... type:"gaze" ...)` 那行**之后**加一行，缓存到本地供上传：
```javascript
    gazeBuffer.push({ t, looking });
```

把 `stopRecordingAndTranscribe` 替换为：
```javascript
async function stopRecordingAndTranscribe() {
  const done = new Promise((res) => (recorder.onstop = res));
  recorder.stop();
  await done;
  const blob = new Blob(audioChunks, { type: "audio/webm" });
  audioChunks = [];
  const form = new FormData();
  form.append("audio", blob, "answer.webm");
  form.append("session_id", sessionId || String(Date.now()));
  form.append("question", document.getElementById("question").textContent || "");
  form.append("gaze", JSON.stringify(gazeBuffer));
  const resp = await fetch("/transcribe", { method: "POST", body: form });
  const data = await resp.json();
  document.getElementById("transcript").textContent = data.summary.transcript || "(无转录)";
  document.getElementById("stats").textContent = JSON.stringify(data.summary, null, 2);
  log("已汇总落库: " + data.saved_to);
}
```

在 `btn-start` 的 onclick 里，`startRecording()` 之前加：
```javascript
  sessionId = String(Date.now());
  gazeBuffer = [];
```

在 `frontend/index.html` 的 `#capture` section 里加：
```html
    <h3>统计量</h3>
    <pre id="stats"></pre>
```

- [ ] **Step 7: 手动验证（清单 Step 6 跑通标志 = L0 完成）**

走完一轮：开始 → 回答 → 结束。
Expected:
- `#stats` 显示一份统计量（transcript、gaze_on_camera_ratio、avg_wpm、flags）。
- `results/session_<id>.json` 文件存在。

- [ ] **Step 8: 红线人工确认（必须亲自做）**

Run:
```bash
find "/Users/kong/Desktop/Glasgow毕业论文/L0-foundation" \
  \( -name "*.webm" -o -name "*.wav" -o -name "*.mp4" -o -name "*.mov" -o -name "*.m4a" -o -name "*.ogg" \) \
  -not -path "*/.venv/*" -not -path "*/node_modules/*" | grep . && echo "❌ 发现音视频文件！" || echo "✅ 磁盘上无任何音视频文件"
```
Expected: `✅ 磁盘上无任何音视频文件`

- [ ] **Step 9: 消融开关手动验证**

把 `server/config.py` 的 `MULTIMODAL_ENABLED = False`，重启服务，再走一轮。
Expected: `#stats` 只剩 transcript + flags，**没有** gaze/wpm 字段。验证后改回 `True`。

- [ ] **Step 10: Commit**

```bash
git add server/session.py server/app.py frontend/ tests/test_session.py
git commit -m "feat(step6): 汇总统计量+落库JSON+no-media红线断言, L0核心完成"
```

---

## Task 7 (清单 Step 7): 部署上线（cloudflared 隧道，别人能用）

**Files:**
- Create: `run.sh`, `tunnel.sh`, `README.md`

- [ ] **Step 1: 写本地启动脚本**

Create `run.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000
```
Run: `chmod +x run.sh`

- [ ] **Step 2: 装 cloudflared 并写隧道脚本**

Run: `brew list cloudflared >/dev/null 2>&1 || brew install cloudflared`

Create `tunnel.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
# 需先另开一个终端跑 ./run.sh（本地 8000），再跑本脚本拿公网链接
exec cloudflared tunnel --url http://localhost:8000
```
Run: `chmod +x tunnel.sh`

> ⚠ 浏览器媒体 API（摄像头/麦克风）要求 HTTPS 或 localhost。cloudflared 给的是 `https://xxx.trycloudflare.com`，满足要求，别人点链接即可授权摄像头。

- [ ] **Step 3: 写 README（含数据安全说明，对齐伦理清单）**

Create `README.md`:
```markdown
# L0 · 面试地基（Foundation）

系统问一道面试题 → 用户对着摄像头回答 → 输出统计量（转录 + 凝视 + 语速）→ **不存任何音视频**。

## 架构
- 浏览器（纯 HTML/JS）：取摄像头/麦克风、浏览器内 MediaPipe 算凝视代理、录音。**原始视频永不离开浏览器。**
- 服务器（FastAPI）：本地 faster-whisper 转录、算 WPM、合并凝视、汇总落库 JSON。**音频转完即丢。**

## 跑起来
```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install ffmpeg cloudflared
./run.sh                      # 本地 http://localhost:8000
# 另开一个终端：
./tunnel.sh                   # 拿到 https://xxx.trycloudflare.com 公网链接
```

## 逐步验证
按 `docs/superpowers/plans/2026-07-12-L0-foundation.md` 的 Step 1→7，每步都有"跑通标志"。

## 消融开关（server/config.py）
- `RAG_ENABLED`：L0 恒 False（占位，L1 接 RAG）。
- `MULTIMODAL_ENABLED`：True=统计量含凝视+语速；False=只有转录（纯文本）。对应 RQ2b。
- 注：这是**消融开关**，与 L2 的 **HUD 显示开关**是两回事，勿混。

## 数据安全（伦理承诺）
- **只存统计量**：`results/session_<id>.json` 里仅有转录文本、凝视占比、平均 WPM、flags。
- **绝不存音视频**：视频不出浏览器；音频服务器转完即丢，不落盘。每轮落库后 `assert_no_media()` 自动自检。
- **`results/` 是被试的研究数据**，已加入 `.gitignore`，**绝不能进 git 仓库**。
- 存储安全：`results/` 仅存于运行主机本地；正式用户研究前应放到受控目录（可加访问控制/加密/匿名化 session_id）。对齐伦理清单"数据安全存储"一条。

## 测试
```bash
.venv/bin/python -m pytest
```
```

- [ ] **Step 4: 全量单测回归**

Run: `.venv/bin/python -m pytest`
Expected: 所有测试 PASS。

- [ ] **Step 5: 手动验证（清单 Step 7 跑通标志 = L0 真正终点）**

一个终端 `./run.sh`，另一个 `./tunnel.sh` 拿到公网链接 → 发给一位同学 → 他在**自己电脑**打开链接 → 授权摄像头 → 答一题 → 你在 `results/` 拿到他的统计量。
Expected: 他没装任何东西，你拿到他的统计量 JSON。

- [ ] **Step 6: Commit**

```bash
git add run.sh tunnel.sh README.md
git commit -m "feat(step7): 本地启动+cloudflared公网隧道+README(含数据安全说明)"
```

---

## 完成标准（L0 Definition of Done）

- [ ] Step 1~7 每个"跑通标志"都手动确认过。
- [ ] `.venv/bin/python -m pytest` 全绿。
- [ ] 走完一轮能拿到统计量 JSON；`find` 确认磁盘上无任何音视频文件。
- [ ] 两个消融开关都在 `config.py`，`MULTIMODAL_ENABLED=False` 时输出退化为纯文本已验证。
- [ ] 别人用自己电脑经公网链接答完一题，你拿到其统计量。
- [ ] README 写明数据安全，`results/` 已被 `.gitignore` 排除。
