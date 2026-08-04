const log = (m) => {
  document.getElementById("log").textContent += m + "\n";
};

const debugMode = new URLSearchParams(location.search).get("debug") === "1";
if (debugMode) {
  document.getElementById("debug-tools").hidden = false;
  document.getElementById("calibration-tools").hidden = false;
  document.getElementById("ws-status").hidden = false;
}

const STUDY_PLAN = window.STUDY_PLAN;
if (!Array.isArray(STUDY_PLAN) || !STUDY_PLAN.length) {
  throw new Error("STUDY_PLAN is missing or empty");
}

// https 页面必须用 wss(否则浏览器拦截混合内容, 抛异常会中断整个脚本)
const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
const ws = new WebSocket(`${wsProto}//${location.host}/ws`);
ws.onopen = () => { document.getElementById("ws-status").textContent = "Connected"; };
ws.onclose = () => { document.getElementById("ws-status").textContent = "Disconnected"; };
ws.onmessage = (e) => { log("收到: " + e.data); };

document.getElementById("btn-hello").onclick = () => {
  ws.send(JSON.stringify({ type: "hello", text: "hello" }));
  log("发送: hello");
};

// ---- Step 2: 取摄像头 + 麦克风（只取流，绝不写文件/落库）----
let mediaStream = null;

async function initMedia() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
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
  initMedia().catch((e) => log("Media permission failed: " + e));

// ---- Step 3: 录音(内存) → 结束时上传转录 ----
let recorder = null;
let audioChunks = [];
let recMime = "audio/webm";

// 本轮 session 状态（Step 6）
let sessionId = null;
let currentQ = 0;
let gazeBuffer = [];

function pickMime() {
  const cands = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
  for (const m of cands) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  }
  return ""; // 让浏览器用默认
}

function startRecording() {
  if (!mediaStream) { log("Enable the camera and microphone first"); return false; }
  audioChunks = [];
  // 只用音频轨道单独建流：避免把 视频+音频 混合流喂给纯音频容器导致录不出数据
  const audioTracks = mediaStream.getAudioTracks();
  if (!audioTracks.length) { log("No audio track is available"); return false; }
  const audioStream = new MediaStream(audioTracks);
  recMime = pickMime();
  recorder = recMime
    ? new MediaRecorder(audioStream, { mimeType: recMime })
    : new MediaRecorder(audioStream);
  recorder.ondataavailable = (e) => { if (e.data && e.data.size) audioChunks.push(e.data); };
  recorder.start(1000); // 每秒切一块，避免只在 stop 才拿到数据
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
  audioChunks = []; // 立即丢弃内存音频
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
  form.append("question", STUDY_PLAN[currentQ].question);
  form.append("rag", STUDY_PLAN[currentQ].rag);
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
  } catch (e) {
    log("Upload failed: " + e);
    showFeedbackFailure();
  }
  showNextButton();
}

// ---- L1 Step 3: participant feedback states ----
function clearFeedback() {
  const area = document.getElementById("feedback");
  area.className = "";
  area.replaceChildren();
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

// 录音/转录由「开始/结束」按钮驱动（见 Step 5）

// ---- Step 4: 浏览器内 MediaPipe 算凝视代理，样本经 WS 发出（为 L2 HUD 预留链路）----
// 阈值单一真源在 server/config.py，页面加载时从 /config 拉取，前端不硬编码。
let GAZE_ON_CAMERA_DEG = 15.0; // 拉取前的兜底默认
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

// 从 4x4 变换矩阵估计头部 yaw/pitch（度），作为"看镜头"代理
function headAnglesDeg(matrix) {
  const m = matrix; // column-major 长度16
  const yaw = Math.atan2(m[8], m[10]) * 180 / Math.PI;
  const pitch = Math.atan2(-m[9], Math.sqrt(m[8] ** 2 + m[10] ** 2)) * 180 / Math.PI;
  return { yaw, pitch };
}

function startGazeLoop() {
  if (gazeTimer) clearInterval(gazeTimer); // 防重入，避免重复 interval
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
    gazeBuffer.push({ t, looking }); // 本地缓存，结束时随音频一起上传
  }, 100); // 10 fps 足够
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

// ---- Step 4 校准：采集两种情况的 yaw/pitch 分布，为阈值提供依据(论文方法章节用)----
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

// ---- L1 Step 3: 问题循环。每题仍保持 TTS 念完再录音，避免题目声音污染答案。----
function beginAnswer() {
  if (recorder && recorder.state === "recording") return; // 防重入
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
    u.onstart = () => { started = true; };  // TTS 真的在念
    u.onend = beginAnswer;                   // 念完再录音(不论题目多长都准确)
    u.onerror = beginAnswer;
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
    // 兜底：仅当 TTS 根本没开始念(静默失败)时才直接开始，不会打断正常念题
    setTimeout(() => { if (!started) beginAnswer(); }, 1000);
  } catch (e) {
    log("TTS unavailable; starting answer directly: " + e);
    beginAnswer();
  }
}

// 收到题目时朗读并显示，念完再开始录音
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
  const item = STUDY_PLAN[currentQ];
  document.getElementById("question-progress").textContent =
    `Question ${currentQ + 1} of ${STUDY_PLAN.length}`;
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
  if (currentQ < STUDY_PLAN.length) {
    askCurrentQuestion();
  } else {
    finishSession();
  }
};
