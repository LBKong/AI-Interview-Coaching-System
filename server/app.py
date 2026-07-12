from pathlib import Path

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from server.asr import AudioChunk, transcribe
from server.metrics import GazeSample
from server.session import pick_question

app = FastAPI(title="L0 Foundation")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    # 本连接内累积凝视样本。L0 汇总实际走 HTTP(见 /transcribe)，
    # 此 WS 通道为 L2 实时 HUD 预留（设计文档 §5）。
    gaze_samples: list[GazeSample] = []
    ws.state.gaze_samples = gaze_samples
    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")
            if mtype == "hello":
                await ws.send_json({"type": "echo", "text": "hi"})
            elif mtype == "start":
                question = pick_question()
                ws.state.question = question
                await ws.send_json({"type": "question", "text": question})
            elif mtype == "gaze":
                gaze_samples.append(GazeSample(t=msg["t"], looking=bool(msg["looking"])))
            else:
                await ws.send_json({"type": "echo", "text": msg.get("text", "")})
    except WebSocketDisconnect:
        return


@app.post("/transcribe")
async def transcribe_endpoint(audio: UploadFile):
    raw = await audio.read()
    if not raw:
        return {"text": "", "words": [], "error": "收到空音频（前端未录到数据）"}
    try:
        # L0：整段音频包成单个 chunk 的迭代器（接口已是流式形状）
        result = transcribe(iter([AudioChunk(data=raw, t_start=0.0)]))
    except Exception as e:  # 返回 JSON 而非 500，便于前端显示真实原因
        print(f"[/transcribe] 转录失败: {e}")
        return {"text": "", "words": [], "error": str(e)}
    return {"text": result.text, "words": [w.__dict__ for w in result.words]}


# 静态前端挂在最后，避免盖过 API 路由
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
