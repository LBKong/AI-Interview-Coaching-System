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

// ---- Step 3: 录音(内存) → 结束时上传转录 ----
let recorder = null;
let audioChunks = [];
let recMime = "audio/webm";

// 本轮 session 状态（Step 6）
let sessionId = null;
let gazeBuffer = [];

function pickMime() {
  const cands = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
  for (const m of cands) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  }
  return ""; // 让浏览器用默认
}

function startRecording() {
  if (!mediaStream) { log("请先开启摄像头+麦克风"); return; }
  audioChunks = [];
  // 只用音频轨道单独建流：避免把 视频+音频 混合流喂给纯音频容器导致录不出数据
  const audioTracks = mediaStream.getAudioTracks();
  if (!audioTracks.length) { log("❌ 没有音频轨道"); return; }
  const audioStream = new MediaStream(audioTracks);
  recMime = pickMime();
  recorder = recMime
    ? new MediaRecorder(audioStream, { mimeType: recMime })
    : new MediaRecorder(audioStream);
  recorder.ondataavailable = (e) => { if (e.data && e.data.size) audioChunks.push(e.data); };
  recorder.start(1000); // 每秒切一块，避免只在 stop 才拿到数据
  log("开始录音");
}

async function stopRecordingAndTranscribe() {
  if (!recorder) { log("尚未开始录音"); return; }
  const done = new Promise((res) => (recorder.onstop = res));
  recorder.stop();
  await done;
  const blob = new Blob(audioChunks, { type: recorder.mimeType || "audio/webm" });
  audioChunks = []; // 立即丢弃内存音频
  log("录到音频: " + blob.size + " 字节");
  if (!blob.size) { log("❌ 录到 0 字节，检查麦克风是否有信号"); return; }
  const form = new FormData();
  form.append("audio", blob, "answer");
  form.append("session_id", sessionId || String(Date.now()));
  form.append("question", document.getElementById("question").textContent || "");
  form.append("gaze", JSON.stringify(gazeBuffer));
  const resp = await fetch("/transcribe", { method: "POST", body: form });
  const data = await resp.json();
  if (data.error) { log("❌ 后端: " + data.error); return; }
  document.getElementById("transcript").textContent = data.summary.transcript || "(无转录)";
  document.getElementById("stats").textContent = JSON.stringify(data.summary, null, 2);
  log("已汇总落库: " + data.saved_to);
}

// 录音/转录由「开始/结束」按钮驱动（见 Step 5）

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
        `${looking ? "看镜头" : "看别处"} (yaw=${yaw.toFixed(0)}, pitch=${pitch.toFixed(0)})`;
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
  initGaze().then(startGazeLoop).catch((e) => log("MediaPipe 失败: " + e));

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
  document.getElementById("stats").textContent = "";
  sessionId = String(Date.now());
  gazeBuffer = [];
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
