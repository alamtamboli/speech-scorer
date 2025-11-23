# backend/app/audio_processor.py
import whisper_timestamped as whisper

import librosa
import numpy as np

FILLER_WORDS = ["um", "uh", "like", "you know", "so", "actually"]

model = whisper.load_model("small")


def transcribe_audio(path: str):
    """Transcribes audio + extracts audio features"""
    result = model.transcribe(path)
    transcript = result.get("text", "").strip()

    y, sr = librosa.load(path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    words = transcript.split()
    wpm = (len(words) / max(duration, 1)) * 60

    frame_length = int(0.02 * sr)
    hop_length = int(0.01 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    pause_ratio = float(np.sum(rms < 0.01)) / max(len(rms), 1)

    filler_count = {f: transcript.lower().count(f) for f in FILLER_WORDS}
    filler_rate = sum(filler_count.values()) / max(len(words), 1)

    audio_stats = {
        "duration_sec": duration,
        "wpm": wpm,
        "pause_ratio": pause_ratio,
        "filler_counts": filler_count,
        "filler_rate_raw": filler_rate
    }

    return transcript, audio_stats
