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
