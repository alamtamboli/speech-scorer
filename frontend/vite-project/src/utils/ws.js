// frontend/vite-project/src/utils/ws.js
export function createSocket(wsUrl = "ws://127.0.0.1:8000/ws/score") {
  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("WebSocket connected");
  };

  ws.onclose = () => {
    console.log("WebSocket closed");
  };

  ws.onerror = (e) => {
    console.error("WebSocket error", e);
  };

  return ws;
}
