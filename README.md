# Multimodal AI Interview Feedback System

This repository contains the research prototype developed for an MSc dissertation on AI-generated interview coaching. The application records spoken interview answers, transcribes them locally, derives limited nonverbal measurements, and generates structured written feedback. It also provides participant questionnaires, condition-blind LLM evaluation, and expert-calibration tools.

The application is a coaching and research system. It is not designed to make recruitment decisions and does not assign candidates scores, grades, or hiring recommendations.

## Research Questions

The project investigates three questions:

- **RQ1:** Can an LLM judge, calibrated against human experts, evaluate AI-generated interview feedback reliably?
- **RQ2a:** Does retrieval-augmented generation (RAG) improve participants' perceived feedback quality?
- **RQ2b:** Does adding nonverbal delivery information improve generated feedback quality?

## Main Features

- A browser-based, four-question interview session.
- Local speech transcription with faster-whisper.
- Word-level timestamps and speaking-rate calculation.
- Browser-side estimation of on-camera gaze using MediaPipe.
- Source-attributed interview guidance retrieved with sentence embeddings and FAISS.
- Structured feedback generation through the Gemini API.
- Counterbalanced RAG conditions for the participant study.
- A seven-item post-feedback questionnaire with an optional comment.
- A condition-blind LLM judge using a five-dimension rubric.
- Calibration of judge scores against human expert ratings.
- Privacy safeguards that prevent raw audio or video from being retained as research data.

## System Architecture

The system is organised into three layers.

### Capture

Participants answer interview questions in the browser. Audio is recorded in memory and uploaded when an answer is submitted. Video frames remain in the browser, where MediaPipe estimates head orientation as an on-camera gaze proxy.

The server uses faster-whisper to produce the transcript and word-level timestamps. The timestamps are used to calculate speaking rate in words per minute (WPM). When multimodal processing is enabled, the response summary also includes the on-camera gaze ratio.

### Generation

The generation layer combines the interview question and transcript with retrieved guidance and delivery measurements when the relevant conditions are enabled. Gemini then produces a structured coaching report.

The coaching structure and rules remain fixed across RAG conditions. RAG changes only whether retrieved interview guidance is supplied to the model. The multimodal switch changes only whether speaking-rate and gaze measurements are available.

### Evaluation

Participants rate each report immediately after reading it. An offline LLM judge can score saved feedback using a five-dimension rubric, while human experts can rate the same reports for judge calibration.

## Participant Flow and Experimental Conditions

A participant session proceeds as follows:

1. Open the study link and grant camera and microphone access.
2. Start the interview and listen to the first question.
3. Answer aloud and submit the recording.
4. Read the generated feedback.
5. Complete the seven-item questionnaire.
6. Continue until all four questions are complete.

Recording begins only after the browser has finished reading the question aloud, preventing the question audio from being included in the answer.

The `group` URL parameter selects one of two counterbalanced RAG plans. Both groups receive the same questions in the same order, but the RAG sequence is reversed.

| Group | Question 0 | Question 1 | Question 2 | Question 3 |
|---|---|---|---|---|
| A | RAG on | RAG off | RAG on | RAG off |
| B | RAG off | RAG on | RAG off | RAG on |

Example local links are:

```text
http://localhost:8000/?group=A
http://localhost:8000/?group=B
```

The application defaults to Group A when the parameter is missing or invalid. Development controls are hidden from ordinary participants and can be displayed with `debug=1`.

## Nonverbal Measurements

### Speaking rate

Speaking rate is calculated from faster-whisper word timestamps. The number of recognised words is divided by the elapsed time between the beginning of the first word and the end of the final word.

A value of zero is returned when fewer than two timestamped words are available or when the duration is not positive. In these cases, the generator treats speaking rate as unavailable delivery evidence.

### On-camera gaze ratio

MediaPipe runs in the browser and estimates head orientation from sampled video frames. A frame is classified as on-camera when the absolute yaw and pitch values are below the threshold in `server/config.py`.

A five-sample majority vote smooths the frame classifications before the final ratio is calculated. This is a head-orientation proxy rather than precise eye tracking.

## Retrieval and Feedback Generation

The RAG pipeline is implemented in `server/rag.py`. Source-attributed interview guidance is stored under `server/knowledge/`, with provenance documented in `server/knowledge/SOURCES.md`.

Retrieval uses:

- `all-MiniLM-L6-v2` sentence embeddings.
- Normalised vectors and inner-product similarity.
- A FAISS vector index.
- The three most relevant chunks by default.

The participant's answer is used as the retrieval query because question context is already included in the question-specific knowledge chunks. The generated FAISS index and chunk cache are reproducible artefacts, are excluded from Git, and are rebuilt automatically when absent.

Feedback generation is implemented in `server/feedback.py` using the pinned model `gemini-3.6-flash`. The report has five possible sections:

1. `Did you answer the question?`
2. `What worked`
3. `What to improve`
4. `Delivery`
5. `Next time`

`What worked` may be omitted when the answer contains no genuine strength. `Delivery` appears only when speaking-rate or gaze information is available. The prompt requires specific, actionable feedback, prohibits invented praise and candidate scoring, and limits the combined number of strengths and improvement points.

Empty or truncated model outputs are rejected. Temporary rate-limit, server, and network failures may be retried, with no more than three attempts. If generation fails, the transcript and derived measurements remain stored, but the feedback is explicitly marked as failed rather than treated as a valid observation.

## Post-feedback Questionnaire

Participants rate seven statements from 1 (`Strongly disagree`) to 7 (`Strongly agree`) and may add an optional comment.

| Item | Construct | Statement |
|---|---|---|
| Q1 | Trust | I trust this feedback. |
| Q2 | Accuracy | This feedback seemed accurate to me. |
| Q3 | Genericity | This feedback was generic — it could have applied to almost anyone's answer. |
| Q4 | Specificity | This feedback was specific to what I actually said. |
| Q5 | Usefulness | This feedback was useful to me. |
| Q6 | Improvement value | This feedback would help me improve my answer. |
| Q7 | Intention to act | I would act on this feedback in a future interview. |

Q3 is intentionally reverse-worded. Its raw response is stored unchanged and can be recoded as `8 - x` during analysis. Questionnaire responses are attached to the same per-question record as the corresponding transcript and feedback.

## LLM Judge and Expert Calibration

The offline judge in `server/judge.py` scores feedback from 1 to 5 on accuracy, specificity, actionability, coverage, and overall usefulness. It receives only the interview question, transcript, and feedback text; experimental flags, retrieved knowledge, and delivery measurements are withheld.

The judge uses a temperature of `0.0`. All five dimensions must be present and contain integer scores from 1 to 5. Malformed responses raise errors rather than receiving default values. The generator and judge use the same pinned model, creating a possible self-preference risk that is examined through expert calibration.

To score valid records from `results/` and write separate files under `scores/`:

```bash
.venv/bin/python -m server.judge
```

Calibration is implemented in `server/calibration.py`. It compares judge scores with expert ratings made using the same rubric and reports rank correlation, quadratic-weighted kappa, exact agreement, agreement within one point, and variance diagnostics.

Expert CSV files must contain the following columns:

```text
sample_id,accuracy,specificity,actionability,coverage,overall_usefulness
```

Run calibration with:

```bash
.venv/bin/python -m server.calibration
```

Judge scores are stored separately and are never written back into participant records.

## Privacy and Data Handling

Privacy is treated as a design constraint.

- Raw video frames are processed in the browser and are never uploaded or stored.
- Audio is uploaded only for local transcription and is not retained as research data.
- The temporary file required by ffmpeg is deleted in a cleanup block, including when decoding fails.
- Every production save operation checks the results directory for audio or video files and raises an error if any are found.
- Records use an opaque session identifier rather than a participant's name or contact details.

Each per-question record may contain the interview question, transcript, condition flags, derived measurements, generated feedback, model metadata, questionnaire responses, and an optional comment. Records use the following naming pattern:

```text
results/session_{session_id}_q{question_index}.json
```

The `results/` directory contains participant research data, remains local to the running system, and is excluded from Git.

## Repository Structure

```text
L0-foundation/
├── frontend/
│   ├── app.js                  # Participant flow, recording, gaze and questionnaire
│   ├── index.html              # Participant interface
│   ├── study_config.js         # Questions and counterbalanced RAG plans
│   └── style.css               # Interface styling
├── server/
│   ├── app.py                  # FastAPI application and routes
│   ├── asr.py                  # Audio decoding and local transcription
│   ├── calibration.py          # Expert-judge calibration
│   ├── config.py               # Models, thresholds and experiment settings
│   ├── feedback.py             # Prompt construction and feedback generation
│   ├── judge.py                # Condition-blind LLM evaluation
│   ├── metrics.py              # Speaking-rate and gaze aggregation
│   ├── rag.py                  # Embedding, indexing and retrieval
│   ├── session.py              # Persistence and privacy auditing
│   └── knowledge/              # Interview guidance and source records
├── tests/                      # Mocked automated tests
├── expert_scores/              # Expert-rating format documentation
├── requirements.txt            # Pinned Python dependencies
├── pytest.ini                  # Test configuration
├── run.sh                      # Cached/offline startup
├── tunnel.sh                   # Optional Cloudflare quick tunnel
└── .python-version             # Python 3.12
```

## Requirements and Installation

The recorded project environment uses Python 3.12. The system also requires ffmpeg, Git, a modern browser with camera and microphone support, internet access for Gemini requests and initial model downloads, and a Gemini API key.

Install the system packages on Ubuntu 24.04:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip ffmpeg git curl
```

On macOS, the equivalent dependencies can be installed with:

```bash
brew install python@3.12 ffmpeg
```

Create the Python environment from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## Gemini Configuration

The application reads `GEMINI_API_KEY` from `.env` in the repository root or from the process environment. Create the file locally with a text editor:

```text
GEMINI_API_KEY=your_key_here
```

Do not commit the real key or include it in documentation, screenshots, or shell history. `.env` is excluded by `.gitignore`.

The configured generator and judge models are defined in `server/config.py` and are pinned to `gemini-3.6-flash` for the recorded study configuration.

## Running the Application

For a fresh installation, allow the models to download during first use and start the server directly:

```bash
.venv/bin/uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Then open `http://localhost:8000/?group=A`. The first transcription downloads the faster-whisper `small.en` model, while the first RAG request downloads `all-MiniLM-L6-v2`.

After the Hugging Face models have been cached, the local offline startup script can be used:

```bash
./run.sh
```

`run.sh` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so it will fail on a fresh machine until the required models have been downloaded.

Camera and microphone access require `localhost` or HTTPS. For remote testing, run the server on `0.0.0.0` and start the optional Cloudflare quick tunnel in a second terminal:

```bash
.venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000
./tunnel.sh
```

The tunnel is suitable for temporary testing rather than permanent deployment.

## API Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the participant interface |
| `/config` | GET | Returns the gaze threshold used by the browser |
| `/ws` | WebSocket | Provides connection checks and reserved real-time messages |
| `/transcribe` | POST | Processes an answer, generates feedback and saves the record |
| `/questionnaire` | POST | Adds questionnaire responses to the matching record |

Important experimental and model settings are held in `server/config.py`. Per-question RAG values come from `frontend/study_config.js` and override the global fallback, ensuring that each record stores the condition actually applied.

## Testing

Network-facing behaviour is mocked in the automated suite, so the tests should not make paid Gemini calls. Run all tests from the repository root:

```bash
.venv/bin/python -m pytest -v
```

The suite covers transcription interfaces, temporary-file cleanup, speaking-rate calculation, gaze smoothing, retrieval, ablation switches, feedback generation, retries, judge blindness, score parsing, questionnaire validation, persistence, privacy auditing, calibration statistics, HTTP routes, and WebSocket behaviour.

Camera permissions, microphone recording, MediaPipe tracking, model downloads, Gemini connectivity, and HTTPS access require manual end-to-end testing.

## Files Excluded from the Repository

The following must remain outside the source repository:

```text
.env
results/
scores/
expert_scores/*.csv
calibration_report.json
.venv/
model caches
__pycache__/
*.pyc
*.wav
*.webm
*.mp4
```

Participant records, judge scores, expert ratings, and calibration outputs are research data or derived research artefacts and are not part of the source-code submission.

## Known Limitations

- On-camera gaze is a head-orientation proxy rather than precise eye tracking.
- Speaking rate and gaze are coarse summary measurements.
- Feedback generation and judging depend on an external Gemini service.
- Using the same model for generation and judging creates a possible self-preference risk.
- RAG supplies relevant source material but cannot guarantee that every generated statement is supported by it.
- The questionnaire was adapted for this study and is not a separately validated psychometric scale.
- The application is a research prototype rather than a production recruitment system.

## Dissertation Context

This repository provides the application source code and its core evaluation modules. The dissertation contains the literature review, experimental rationale, results, interpretation, and limitations. Participant research data and generated evaluation outputs are intentionally stored separately from this repository.
