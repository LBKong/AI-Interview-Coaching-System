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
