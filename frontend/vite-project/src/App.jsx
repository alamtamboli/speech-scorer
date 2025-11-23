import React, { useState, useEffect, useRef } from "react";
import ScorePanel from "./components/ScorePanel";
import AudioUploader from "./components/AudioUploader";
import { createSocket } from "./utils/ws";
import "./styles.css";

export default function App() {
  const [transcript, setTranscript] = useState("");
  const [result, setResult] = useState(null);
  const [quick, setQuick] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
   wsRef.current = createSocket("ws://127.0.0.1:8000/ws/score");

    wsRef.current.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "quick") setQuick(msg.data);
      if (msg.type === "final") setResult(msg.data);
    };

    return () => wsRef.current.close();
  }, []);

  const handleRealtimeScore = () => {
    wsRef.current.send(JSON.stringify({ transcript }));
  };

  const handleBatchScore = async () => {
   const res = await fetch("http://127.0.0.1:8000/score", {

      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript }),
    });
    setResult(await res.json());
  };

  return (
    <div className="app">
      <main>
        <div className="left">
          <textarea
            placeholder="Paste or type transcript..."
            rows="10"
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
          />
          <button onClick={handleRealtimeScore}>Score Realtime</button>
          <button onClick={handleBatchScore}>Score Once</button>

          <AudioUploader setResult={setResult} />
        </div>

        <div className="right">
          <ScorePanel quick={quick} result={result} />
        </div>
      </main>
    </div>
  );
}
