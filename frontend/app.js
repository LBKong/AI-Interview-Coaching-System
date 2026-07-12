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
  const resp = await fetch("/transcribe", { method: "POST", body: form });
  const data = await resp.json();
  if (data.error) { log("❌ 后端: " + data.error); return; }
  document.getElementById("transcript").textContent = data.text || "(无转录)";
  log("转录完成");
}

document.getElementById("btn-rec-start").onclick = startRecording;
document.getElementById("btn-rec-stop").onclick = () =>
  stopRecordingAndTranscribe().catch((e) => log("转录失败: " + e));
