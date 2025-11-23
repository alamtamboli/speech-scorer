# Speech Scorer – AI-Powered Speech Evaluation System

[![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-blue)]()
[![Backend](https://img.shields.io/badge/Backend-FastAPI-green)]()
[![AI](https://img.shields.io/badge/AI-NLP%20%7C%20Speech%20Analysis-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

A production-grade **AI-powered Speech Evaluation System** that analyzes self‑introductions using advanced **NLP, semantic similarity, sentiment detection, Whisper transcription, Neo4j relevance analysis**

---

## 🚀 Features

### 🎙️ Speech Analysis
- Whisper‑based transcription  
- Words‑Per‑Minute (WPM) detection  
- Filler word analysis  
- Grammar error rate (LanguageTool)  
- Sentiment positivity (VADER)  
- Vocabulary richness (TTR)  
- Keyword relevance scoring  
- Logical order & flow detection  
- Semantic similarity scoring (Sentence Transformers)  
- Neo4j‑based conceptual relevance  

---

## 🎨 Frontend (React + Vite)
- Clean modern UI  
- Real‑time scoring via WebSocket  
- Audio upload support  
- Radar chart visualization  
- Detailed per‑criterion breakdown  
- Friendly error handling  

---

## 🏗️ Architecture Overview

```
React Frontend ↔ FastAPI Backend ↔ AI Models
                              ↳ Whisper (speech)
                              ↳ Sentence Transformers
                              ↳ Neo4j Knowledge Graph
                              ↳ LanguageTool
                              ↳ Sentiment Analyzer
```

---

## 📂 Project Structure

```
speech-scorer/
│── backend/
│   └── app/
│       ├── main.py
│       ├── score_engine.py
│       ├── audio_processor.py
│       ├── neo4j_layer.py
│       ├── groq_client.py
│       ├── ollama_client.py
│       ├── rubric.json
│
│── frontend/
│   └── vite-project/
│       ├── src/
│       │   ├── App.jsx
│       │   ├── components/
│       │   ├── utils/
│       │   ├── styles.css
│
└── README.md
```

---



### Backend Setup
```bash
cd backend
python -m venv .venv
.\.venv\Scriptsctivate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup
```bash
cd frontend/vite-project
npm install
npm run dev
```

---

## 📡 API Endpoints

### `POST /score`
Scores transcript text.

### `POST /upload_audio`
Upload audio → transcribe → score.

### `POST /rewrite`
AI‑enhanced rewrite of transcript.

### `WebSocket /ws/score`
Real‑time scoring as user types.

---

## 📊 Example Scoring Output

```
{
  "overall": 82.5,
  "per_criteria": [
    {
      "id": "keyword_relevance",
      "score": 0.85,
      "band": "Good",
      "components": { "matched_groups": 4 }
    }
  ]
}
```

---

## 🌍 Deployment Options

### Backend
- Render (free)
- Railway
- AWS EC2 Free Tier
- Local host

### Frontend
- Vercel
- Netlify
- GitHub Pages

---

## 🤝 Contributing
Pull requests are welcome!  
Open an issue for feature discussion.

---

## 📄 License
MIT License © 2025 Speech Scorer Project
