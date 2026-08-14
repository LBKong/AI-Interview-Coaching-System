"""L0 global configuration. Both ablation switches live here from the start and remain here throughout."""
from pathlib import Path

# ---- Two ablation switches (used for L1/L2 ablations) ----
RAG_ENABLED = False          # Always False in L0 as a placeholder; RAG is connected in L1
MULTIMODAL_ENABLED = True    # True=metrics include gaze+speaking rate; False=transcript only (plain text). Corresponds to RQ2b

# ---- Gaze-proxy threshold (must be documented in the thesis methods section) ----
# Classification: "looking at camera" ⇔ |yaw| < threshold and |pitch| < threshold (degrees).
# Calibration basis (2026-07-13, single-participant pilot):
#   Looking at camera: |yaw| max=1.4°, |pitch| max=3.3°; looking away (horizontal head turn): |yaw| p50≈35°.
#   The groups have a ~34° yaw gap; 15° lies in the middle, with ~4.5× margin above the engaged maximum and far below averted.
#   This margin, rather than a smaller value, accommodates natural head movement while answering (gestures/thinking/looking up to recall).
# Note: the pilot's "looking away" condition used a horizontal head turn and did not stress the pitch dimension. To robustly detect
#       vertical diversion such as looking down at a desk, add a vertical calibration. The frontend's threshold-calibration buttons reproduce data collection.
GAZE_ON_CAMERA_DEG = 15.0

# ---- ASR (local faster-whisper) ----
WHISPER_MODEL = "small.en"   # English interviews; use "base.en" for more speed on CPU
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_DEVICE = "cpu"       # Mac CPU; change if a GPU is available

# ---- Question bank (start with a few hand-written questions; no database) ----
QUESTIONS = [
    "Tell me about a challenging project you worked on and your role in it.",
    "Describe a time you had a conflict with a teammate and how you handled it.",
    "Why are you interested in this position?",
    "What is a skill you recently learned, and how did you learn it?",
]

# ---- Persistence directory (metrics only; never audio or video) ----
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ---- LLM feedback generation (Gemini, L1 Step 2) ----
# Put GEMINI_API_KEY in .env (already in .gitignore; the key never enters the repository)
import os

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass  # Fall back to the process environment when python-dotenv is not installed

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Pin the model rather than using a *-latest alias: aliases auto-upgrade and break evaluation reproducibility.
# Choose flash rather than flash-lite: feedback text is this study's dependent variable (expert review / LLM judge / participant questionnaire
# all evaluate it), so generation quality is not downgraded. The 2.5 family retires in 2026-10 and preview models can change without notice; both are avoided.
# Switched from gemini-3.5-flash to gemini-3.6-flash on 2026-07-24: every 3.5-flash call during development
# returned 503 (server-side high demand), making it unreliable on this network/timezone; 3.6-flash is verified end-to-end.
# ⚠ This model is frozen for the study—once evaluation data collection begins, it must never change (or comparability/reproducibility would break).
GEMINI_MODEL = "gemini-3.6-flash"
FEEDBACK_TEMPERATURE = 0.3          # Low temperature: feedback must be stable and reproducible (evaluation is rerun)
FEEDBACK_LANGUAGE = "English"       # Interviews and answers are in English; change to "Chinese" for Chinese feedback

# ---- LLM judge (offline evaluation) ----
# Judge = generation model (both gemini-3.6-flash), a deliberate and known trade-off:
# this account's Pro tier is unavailable/unstable (2.5-pro account-level block, 3-pro-preview retired, and flash-lite changed from
# 200 to 404 within days). Only 3.6-flash has repeatedly proved stable and available. A 404 midway through judge evaluation is worse than
# same-model self-preference risk. Expert calibration tests self-preference (2–3 experts score ~20 samples using the same rubric):
# if the same-model judge correlates highly with experts, self-preference has no material effect; if not, the judge will not be used anyway.
# This is an honest, defensible limitation (document it in the thesis limitations). Therefore judge_model_version must be recorded for every score.
JUDGE_MODEL = "gemini-3.6-flash"    # Same model as generation (reason above); pinned non-alias for reproducibility
JUDGE_TEMPERATURE = 0.0             # Scoring should be as deterministic and reproducible as possible (generation uses 0.3)
