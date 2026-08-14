const log = (m) => {
  document.getElementById("log").textContent += m + "\n";
};

const debugMode = new URLSearchParams(location.search).get("debug") === "1";
if (debugMode) {
  document.getElementById("debug-tools").hidden = false;
  document.getElementById("calibration-tools").hidden = false;
  document.getElementById("ws-status").hidden = false;
}

const params = new URLSearchParams(window.location.search);
const group = (params.get("group") || "A").trim().toUpperCase();
const plan = window.STUDY_PLANS[group] || window.STUDY_PLANS.A;
if (!Array.isArray(plan) || !plan.length) {
  throw new Error("Selected study plan is missing or empty");
}

// HTTPS pages must use wss (otherwise the browser blocks mixed content, and the exception stops the entire script)
const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
const ws = new WebSocket(`${wsProto}//${location.host}/ws`);
ws.onopen = () => { document.getElementById("ws-status").textContent = "Connected"; };
ws.onclose = () => { document.getElementById("ws-status").textContent = "Disconnected"; };
ws.onmessage = (e) => { log("收到: " + e.data); };

document.getElementById("btn-hello").onclick = () => {
  ws.send(JSON.stringify({ type: "hello", text: "hello" }));
  log("发送: hello");
};

// ---- Step 2: Access camera + microphone (stream only; never write files or persist data) ----
let mediaStream = null;

async function initMedia() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  document.getElementById("capture").hidden = false;
  document.getElementById("selfview").srcObject = mediaStream;

  // Microphone-level self-check
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
  initMedia().catch((e) => log("Media permission failed: " + e));

// ---- Step 3: Record (in memory) → upload for transcription when finished ----
let recorder = null;
let audioChunks = [];
let recMime = "audio/webm";

// State for this session (Step 6)
let sessionId = null;
let currentQ = 0;
let gazeBuffer = [];

function pickMime() {
  const cands = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
  for (const m of cands) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  }
  return ""; // Let the browser use its default
}

function startRecording() {
  if (!mediaStream) { log("Enable the camera and microphone first"); return false; }
  audioChunks = [];
  // Build a separate stream from audio tracks only: feeding a video+audio stream to an audio-only container can produce no data
  const audioTracks = mediaStream.getAudioTracks();
  if (!audioTracks.length) { log("No audio track is available"); return false; }
  const audioStream = new MediaStream(audioTracks);
  recMime = pickMime();
  recorder = recMime
    ? new MediaRecorder(audioStream, { mimeType: recMime })
    : new MediaRecorder(audioStream);
  recorder.ondataavailable = (e) => { if (e.data && e.data.size) audioChunks.push(e.data); };
  recorder.start(1000); // Emit one chunk per second instead of receiving data only at stop
  log("Recording started");
  return true;
}

async function stopRecordingAndTranscribe() {
  if (!sessionId) {
    log("ERROR: sessionId is missing; upload stopped");
    showFeedbackFailure();
    showNextButton();
    return;
  }
  if (!recorder || recorder.state !== "recording") {
    log("Recording has not started");
    showFeedbackFailure();
    showNextButton();
    return;
  }
  document.getElementById("btn-end").disabled = true;
  const done = new Promise((res) => (recorder.onstop = res));
  recorder.stop();
  await done;
  const blob = new Blob(audioChunks, { type: recorder.mimeType || "audio/webm" });
  recorder = null;
  audioChunks = []; // Immediately discard in-memory audio
  log("Recorded audio: " + blob.size + " bytes");
  if (!blob.size) {
    log("No audio was recorded");
    showFeedbackFailure();
    showNextButton();
    return;
  }

  showFeedbackGenerating();
  const form = new FormData();
  form.append("audio", blob, "answer");
  form.append("session_id", sessionId);
  form.append("question_index", String(currentQ));
  form.append("question", plan[currentQ].question);
  form.append("rag", plan[currentQ].rag);
  form.append("gaze", JSON.stringify(gazeBuffer));

  try {
    const resp = await fetch("/transcribe", { method: "POST", body: form });
    const data = await resp.json();
    if (data.error || !data.summary) {
      log("Backend error: " + (data.error || "missing summary"));
      showFeedbackFailure();
      showNextButton();
      return;
    }

    document.getElementById("transcript").textContent = data.summary.transcript || "(No transcript)";
    document.getElementById("stats").textContent = JSON.stringify(data.summary, null, 2);
    log("Saved: " + data.saved_to);

    const feedback = data.summary.feedback;
    if (feedback && feedback.status === "ok" && feedback.text) {
      showFeedbackSuccess(feedback.text);
    } else {
      showFeedbackFailure();
    }
    showQuestionnaire();
  } catch (e) {
    log("Upload failed: " + e);
    showFeedbackFailure();
    showNextButton();
  }
}

// ---- L1 Step 3: participant feedback states ----
function clearQuestionnaire() {
  const area = document.getElementById("questionnaire");
  area.hidden = true;
  area.replaceChildren();
}

function clearFeedback() {
  const area = document.getElementById("feedback");
  area.className = "";
  area.replaceChildren();
  clearQuestionnaire();
  document.getElementById("btn-next").hidden = true;
}

function showFeedbackGenerating() {
  const area = document.getElementById("feedback");
  area.className = "feedback-loading";
  area.replaceChildren();
  const spinner = document.createElement("span");
  spinner.className = "spinner";
  spinner.setAttribute("aria-hidden", "true");
  const message = document.createElement("p");
  message.textContent = "Generating your feedback — this takes around 10–15 seconds…";
  area.append(spinner, message);
  document.getElementById("answer-status").textContent = "Your answer has been submitted.";
}

function renderFeedbackText(text) {
  const area = document.getElementById("feedback");
  area.className = "feedback-success";
  area.replaceChildren();
  const fixedHeadings = new Set([
    "Did you answer the question?",
    "What worked",
    "What to improve",
    "Delivery",
    "Next time",
  ]);
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    // The prompt asks for ### headings, but tolerate another Markdown level or
    // a bare fixed heading so harmless model formatting drift cannot break UI.
    const headingMatch = line.match(/^#{1,6}\s+(.+)$/);
    const content = headingMatch ? headingMatch[1] : line;
    const isHeading = Boolean(headingMatch) || fixedHeadings.has(content);
    const node = document.createElement(isHeading ? "h4" : "p");
    node.textContent = content;
    area.appendChild(node);
  }
}

function showFeedbackSuccess(text) {
  renderFeedbackText(text);
  document.getElementById("answer-status").textContent = "Feedback ready.";
}

function showFeedbackFailure() {
  const area = document.getElementById("feedback");
  area.className = "feedback-failure";
  area.replaceChildren();
  const heading = document.createElement("h3");
  heading.textContent = "Feedback unavailable";
  const message = document.createElement("p");
  message.textContent = "The feedback couldn't be generated this time. That's a problem on our side, not with your answer.";
  area.append(heading, message);
  document.getElementById("answer-status").textContent = "Your answer has been submitted.";
}

function showNextButton() {
  document.getElementById("btn-next").hidden = false;
}

// ---- Post-feedback questionnaire: raw 1–7 responses are stored as entered. ----
const QUESTIONNAIRE_ITEMS = [
  "I trust this feedback.",
  "This feedback seemed accurate to me.",
  "This feedback was generic — it could have applied to almost anyone's answer.",
  "This feedback was specific to what I actually said.",
  "This feedback was useful to me.",
  "This feedback would help me improve my answer.",
  "I would act on this feedback in a future interview.",
];

function showQuestionnaire() {
  const area = document.getElementById("questionnaire");
  area.replaceChildren();
  area.hidden = false;

  const form = document.createElement("form");
  form.id = "questionnaire-form";

  const instruction = document.createElement("p");
  instruction.className = "questionnaire-instruction";
  instruction.textContent =
    "How much do you agree with each statement about the feedback you just read?  " +
    "(1 = Strongly disagree, 7 = Strongly agree)";
  form.appendChild(instruction);

  QUESTIONNAIRE_ITEMS.forEach((item, index) => {
    const questionNumber = index + 1;
    const fieldset = document.createElement("fieldset");
    fieldset.className = "likert-item";
    const legend = document.createElement("legend");
    legend.textContent = `${questionNumber}. ${item}`;
    fieldset.appendChild(legend);

    const scale = document.createElement("div");
    scale.className = "likert-scale";
    for (let value = 1; value <= 7; value += 1) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = `q${questionNumber}`;
      input.value = String(value);
      input.required = true;
      label.append(input, String(value));
      scale.appendChild(label);
    }
    fieldset.appendChild(scale);
    form.appendChild(fieldset);
  });

  const commentLabel = document.createElement("label");
  commentLabel.className = "questionnaire-comment";
  commentLabel.textContent = "Anything else about this feedback? (optional)";
  const comment = document.createElement("textarea");
  comment.name = "comment";
  comment.rows = 3;
  commentLabel.appendChild(comment);
  form.appendChild(commentLabel);

  const submit = document.createElement("button");
  submit.type = "submit";
  submit.id = "btn-questionnaire-submit";
  submit.textContent = "Submit & continue";
  submit.disabled = true;
  form.appendChild(submit);

  const status = document.createElement("p");
  status.className = "questionnaire-status";
  status.setAttribute("aria-live", "polite");
  form.appendChild(status);

  form.addEventListener("change", () => {
    submit.disabled = form.querySelectorAll("input[type=radio]:checked").length !== 7;
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    submit.disabled = true;
    status.textContent = "Saving your responses…";
    const responses = new FormData(form);
    responses.append("session_id", sessionId);
    responses.append("question_index", String(currentQ));
    try {
      const response = await fetch("/questionnaire", { method: "POST", body: responses });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || `HTTP ${response.status}`);
      }
      log("Questionnaire saved: " + data.saved_to);
      clearQuestionnaire();
      showNextButton();
    } catch (error) {
      log("Questionnaire save failed: " + error);
      status.textContent = "Your responses could not be saved. Please try again.";
      submit.disabled = false;
    }
  });

  area.appendChild(form);
}

// Recording/transcription is driven by the Start/End buttons (see Step 5)

// ---- Step 4: Compute the gaze proxy in-browser with MediaPipe; send samples over WS (reserved for the L2 HUD) ----
// The threshold's single source of truth is server/config.py; fetch it from /config at page load instead of hard-coding it in the frontend.
let GAZE_ON_CAMERA_DEG = 15.0; // Fallback default before the fetch completes
fetch("/config").then((r) => r.json()).then((c) => {
  if (typeof c.gaze_on_camera_deg === "number") {
    GAZE_ON_CAMERA_DEG = c.gaze_on_camera_deg;
    log("凝视阈值(来自后端): " + GAZE_ON_CAMERA_DEG + "°");
  }
}).catch(() => {});
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

// Estimate head yaw/pitch (degrees) from the 4x4 transform matrix as a "looking at camera" proxy
function headAnglesDeg(matrix) {
  const m = matrix; // column-major, length 16
  const yaw = Math.atan2(m[8], m[10]) * 180 / Math.PI;
  const pitch = Math.atan2(-m[9], Math.sqrt(m[8] ** 2 + m[10] ** 2)) * 180 / Math.PI;
  return { yaw, pitch };
}

function startGazeLoop() {
  if (gazeTimer) clearInterval(gazeTimer); // Prevent re-entry and duplicate intervals
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
        `${looking ? "Looking at camera" : "Looking away"} (yaw=${yaw.toFixed(0)}, pitch=${pitch.toFixed(0)})`;
    }
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "gaze", t, looking }));
    }
    gazeBuffer.push({ t, looking }); // Cache locally and upload with the audio at the end
  }, 100); // 10 fps is sufficient
}

function stopGazeLoop() {
  if (gazeTimer) clearInterval(gazeTimer);
  gazeTimer = null;
}

document.getElementById("btn-gaze").onclick = () =>
  initGaze().then(() => {
    document.getElementById("btn-gaze").textContent = "Gaze tracking ready";
    document.getElementById("btn-gaze").disabled = true;
  }).catch((e) => log("MediaPipe failed: " + e));

// ---- Step 4 calibration: collect yaw/pitch distributions in two conditions to justify the threshold (for the thesis methods section) ----
function pct(arr, p) {
  if (!arr.length) return NaN;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
}

function collectCalibration(label, seconds = 5) {
  if (!faceLandmarker) { log("请先点「开始凝视检测」加载 MediaPipe"); return; }
  const video = document.getElementById("selfview");
  const yaws = [], pitches = [];
  log(`校准[${label}] 开始，请保持 ${seconds} 秒…`);
  const timer = setInterval(() => {
    if (video.readyState < 2) return;
    const res = faceLandmarker.detectForVideo(video, performance.now());
    const mats = res.facialTransformationMatrixes;
    if (mats && mats.length) {
      const { yaw, pitch } = headAnglesDeg(mats[0].data);
      yaws.push(Math.abs(yaw));
      pitches.push(Math.abs(pitch));
    }
  }, 100);
  setTimeout(() => {
    clearInterval(timer);
    const f = (x) => (isNaN(x) ? "NA" : x.toFixed(1));
    log(`校准[${label}] n=${yaws.length}  |yaw| p50/p90/max=${f(pct(yaws,50))}/${f(pct(yaws,90))}/${f(Math.max(...yaws))}` +
        `  |pitch| p50/p90/max=${f(pct(pitches,50))}/${f(pct(pitches,90))}/${f(Math.max(...pitches))}`);
  }, seconds * 1000);
}

document.getElementById("btn-calib-look").onclick = () => collectCalibration("看镜头");
document.getElementById("btn-calib-away").onclick = () => collectCalibration("看别处");

// ---- L1 Step 3: Question loop. For every question, start recording only after TTS finishes so the prompt audio does not contaminate the answer. ----
function beginAnswer() {
  if (recorder && recorder.state === "recording") return; // Prevent re-entry
  if (!startRecording()) return;
  if (faceLandmarker) startGazeLoop();
  document.getElementById("btn-end").hidden = false;
  document.getElementById("btn-end").disabled = false;
  document.getElementById("answer-status").textContent = "Recording — answer the question, then submit.";
  log("Answer started (recording + gaze)");
}

function askAndListen(text) {
  document.getElementById("question").textContent = text;
  try {
    let started = false;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US";
    u.onstart = () => { started = true; };  // TTS has actually started speaking
    u.onend = beginAnswer;                   // Record after speech ends (accurate regardless of question length)
    u.onerror = beginAnswer;
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
    // Fallback: start directly only if TTS never began (silent failure), without interrupting normal speech
    setTimeout(() => { if (!started) beginAnswer(); }, 1000);
  } catch (e) {
    log("TTS unavailable; starting answer directly: " + e);
    beginAnswer();
  }
}

// Read and display a received question, then start recording after it has been spoken
const origOnMessage = ws.onmessage;
ws.onmessage = (e) => {
  origOnMessage(e);
  let msg;
  try { msg = JSON.parse(e.data); } catch { return; }
  if (msg.type === "question") {
    askAndListen(msg.text);
  }
};

function resetQuestionState() {
  speechSynthesis.cancel();
  stopGazeLoop();
  audioChunks = [];
  gazeBuffer = [];
  recorder = null;
  sessionT0 = null;
  document.getElementById("transcript").textContent = "";
  document.getElementById("stats").textContent = "";
  document.getElementById("answer-status").textContent = "Listen to the question. Recording starts after it is read aloud.";
  document.getElementById("btn-end").hidden = true;
  document.getElementById("btn-end").disabled = false;
  clearFeedback();
}

function askCurrentQuestion() {
  resetQuestionState();
  const item = plan[currentQ];
  document.getElementById("question-progress").textContent =
    `Question ${currentQ + 1} of ${plan.length}`;
  document.getElementById("question").textContent = item.question;
  askAndListen(item.question);
  log(`Question ${currentQ + 1} started (RAG ${item.rag})`);
}

function finishSession() {
  speechSynthesis.cancel();
  stopGazeLoop();
  document.getElementById("interview-panel").hidden = true;
  document.getElementById("end-screen").hidden = false;
}

document.getElementById("btn-start").onclick = () => {
  if (!mediaStream) {
    log("Enable the camera and microphone first");
    return;
  }
  sessionId = String(Date.now());
  currentQ = 0;
  document.getElementById("btn-start").hidden = true;
  document.getElementById("end-screen").hidden = true;
  document.getElementById("interview-panel").hidden = false;
  askCurrentQuestion();
  log("Session started; sessionId fixed for all questions");
};

document.getElementById("btn-end").onclick = async () => {
  stopGazeLoop();
  await stopRecordingAndTranscribe();
  log(`Question ${currentQ + 1} submitted`);
};

document.getElementById("btn-next").onclick = () => {
  currentQ += 1;
  if (currentQ < plan.length) {
    askCurrentQuestion();
  } else {
    finishSession();
  }
};
