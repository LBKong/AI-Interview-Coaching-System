from fastapi.testclient import TestClient
from server.app import app


def test_ws_echo_replies_hi_to_hello():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "text": "hello"})
        reply = ws.receive_json()
    assert reply["type"] == "echo"
    assert reply["text"] == "hi"
