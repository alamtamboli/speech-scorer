# backend/app/main.py
import os
import json
import asyncio
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from .score_engine import ScoreEngine
from .audio_processor import transcribe_audio
from .groq_client import groq_complete
from .ollama_client import ollama_generate

app = FastAPI(title="Speech Scorer")

BASE_DIR = os.path.dirname(__file__)
RUBRIC_PATH = os.path.join(BASE_DIR, "rubric.json")
engine = ScoreEngine(rubric_path=RUBRIC_PATH)


class ScoreRequest(BaseModel):
    transcript: str
    include_audio: bool = False


@app.post("/score")
async def score_endpoint(req: ScoreRequest):
    if not req.transcript or not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty")

    result = engine.score_text(req.transcript)

    return {
        "transcript": req.transcript,
        "score_details": result["per_criteria"],
        "overall_score": result["overall"]
    }


@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    tmp_path = f"/tmp/{file.filename}"
    contents = await file.read()

    with open(tmp_path, "wb") as f:
        f.write(contents)

    transcript, audio_stats = transcribe_audio(tmp_path)
    score = engine.score_text(transcript, audio_stats=audio_stats)

    return {
        "transcript": transcript,
        "audio_stats": audio_stats,
        "score_details": score["per_criteria"],
        "overall_score": score["overall"]
    }


@app.post("/rewrite")
async def rewrite_endpoint(payload: dict):
    transcript = payload.get("transcript", "")
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript required")

    prompt = (
        "Rewrite the following self-introduction to improve clarity, confidence, and flow:\n\n"
        f"{transcript}\n\nImproved Version:"
    )

    try:
        if os.getenv("GROQ_API_KEY"):
            rewritten = groq_complete(prompt)
        else:
            rewritten = ollama_generate(prompt)
    except Exception:
        rewritten = ollama_generate(prompt)

    return {"rewritten": rewritten.strip()}


@app.websocket("/ws/score")
async def websocket_score(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            transcript = data.get("transcript", "")

            if not transcript:
                await websocket.send_json({"type": "error", "message": "Transcript missing"})
                continue

            quick_scan = engine.quick_rule_scan(transcript)
            await websocket.send_json({"type": "quick", "data": quick_scan})

            await asyncio.sleep(0.2)

            full_score = engine.score_text(transcript)
            await websocket.send_json({"type": "final", "data": full_score})

    except WebSocketDisconnect:
        return
