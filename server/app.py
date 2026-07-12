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
