"""L0 全局配置。两个消融开关从一开始就在这里，贯穿全程。"""
from pathlib import Path

# ---- 两个消融开关（L1/L2 跑消融要靠它们）----
RAG_ENABLED = False          # L0 恒 False，占位；L1 才接 RAG
MULTIMODAL_ENABLED = True    # True=统计量含凝视+语速；False=只有转录(纯文本)。对应 RQ2b

# ---- 凝视代理阈值（论文方法章节需交代）----
# 判定："看镜头" ⇔ |yaw| < 阈值 且 |pitch| < 阈值（度）。
# 校准依据(2026-07-13, 单被试预试)：
#   看镜头 |yaw| max=1.4°, |pitch| max=3.3°；看别处(水平转头) |yaw| p50≈35°。
#   两组在 yaw 上有 ~34° 空档；15° 居中，高于 engaged 最大值 ~4.5 倍余量、远低于 averted。
#   留余量而非取更小值，是为容忍真实回答时的自然头动(手势/思考/抬头回忆)。
# 注：预试的"看别处"为水平转头，未压 pitch 维度；如需稳健抓"低头看桌面"等垂直移开，
#     应补一次垂直校准。前端页面「校准阈值」按钮可复现采集。
GAZE_ON_CAMERA_DEG = 15.0

# ---- ASR（本地 faster-whisper）----
WHISPER_MODEL = "small.en"   # 英文面试；CPU 上可换 "base.en" 提速
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_DEVICE = "cpu"       # Mac CPU；有 GPU 可改

# ---- 题库（先手写几道，不建题库）----
QUESTIONS = [
    "Tell me about a challenging project you worked on and your role in it.",
    "Describe a time you had a conflict with a teammate and how you handled it.",
    "Why are you interested in this position?",
    "What is a skill you recently learned, and how did you learn it?",
]

# ---- 落库目录（只存统计量，绝不存音视频）----
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ---- LLM 反馈生成（Gemini，L1 Step 2）----
# .env 里放 GEMINI_API_KEY（已在 .gitignore；密钥不进仓库）
import os

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass  # 没装 python-dotenv 时退回读进程环境变量

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# 型号写死、不用 *-latest 别名：别名会自动升级，破坏评估结果的可复现性。
# 选 flash 而非 flash-lite：反馈文本是本研究的因变量（专家评审 / LLM 裁判 / 被试问卷
# 都在评它），生成质量不降档。2.5 系 2026-10 退役、preview 系会无预警变动，均已避开。
# 2026-07-24 从 gemini-3.5-flash 切到 gemini-3.6-flash：3.5-flash 在整个开发期每次调用
# 都返回 503（服务端 high-demand），本网络/时区下不可靠；3.6-flash 已端到端验证可用。
# ⚠ 此型号已为研究冻结——一旦开始采集评估数据，绝不可再改（否则破坏可比性/可复现性）。
GEMINI_MODEL = "gemini-3.6-flash"
FEEDBACK_TEMPERATURE = 0.3          # 低温：反馈要稳、可复现（评估要复算）
FEEDBACK_LANGUAGE = "English"       # 面试与回答都是英文；想看中文反馈改成 "Chinese"

# ---- LLM 裁判（evaluation，离线）----
# 裁判 = 生成模型（都 gemini-3.6-flash），这是刻意且已知的取舍：
# 本账户 Pro 档不可用/不稳定（2.5-pro 账户级屏蔽、3-pro-preview 退役、flash-lite 数天内从
# 200 变 404），唯一反复验证"稳定可用"的只有 3.6-flash。裁判评估中途 404 的风险，比同模型的
# self-preference 风险更糟。self-preference 由专家校准环节检验（2-3 位专家按同一 rubric 打
# ~20 样本）：若同模型裁判与专家高度相关，则 self-preference 无实质影响；若不相关，本就不会用它。
# 这是一条诚实、可辩护的局限（论文 limitations 需写明）。因此 judge_model_version 必须每条都记录。
JUDGE_MODEL = "gemini-3.6-flash"    # 与生成同模型（原因见上）；写死非别名，保证可复现
JUDGE_TEMPERATURE = 0.0             # 打分要尽量确定、可复现（生成侧是 0.3）
