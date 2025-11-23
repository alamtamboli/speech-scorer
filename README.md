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




## 🤝 Contributing
Pull requests are welcome!  
Open an issue for feature discussion.

---

## 📄 License
MIT License © 2025 Speech Scorer Project



 How Scoring Formula Works (Detailed Explanation)

The system generates a final score based on **weighted criteria** defined in `rubric.json`.

The scoring engine evaluates the transcript across 6 major dimensions:

---

## 1️⃣ Situation Level (5%)
Checks if the introduction fits expected structure  
**Formula:**

```
band_score = score_of_matched_band / max_band_score
```

---

## 2️⃣ Keyword Relevance (20%)
Checks if essential introduction elements exist  
(name, education, hobbies, family, goals)

```
matched_groups = count(keyword_groups_matched)
band_score = score_of_best_matching_band / max_band_score
```

---

## 3️⃣ Order & Flow (5%)
Analyzes logical order:  
**Greeting → Name → Education → Family → Experience → Strengths → Hobbies → Goals**

```
if indices == sorted(indices): Correct Order
elif some in order: Partial Order
else: No Order
```

---

## 4️⃣ Transcript Length (10%)
Ideal range = 70–150 words.

```
if wc < min: score = wc/min
if wc > max: score = 1 - ((wc-max)/max)
else: score = 1
```

---

## 5️⃣ WPM – Words Per Minute (10%)
Based on Whisper audio statistics or estimated defaults.

Ranges like:

```
161+    → Too Fast (2 pts)
110-140 → Ideal (10 pts)
<90     → Very Slow (6 pts)
```

---

## 6️⃣ Grammar Error Rate (10%)
Uses **LanguageTool**:

```
error_rate = (errors / word_count) * 100
Find band → scale between 0–10
```

---

## 7️⃣ Vocabulary Richness (TTR) (10%)
Type-token ratio:

```
TTR = unique_words / total_words
Map TTR bands to scores (0–10)
```

---

## 8️⃣ Filler Word Rate (15%)
Counts fillers:

```
['um','uh','like','you know','so']
filler_rate = (filler_count / total_words) * 100
```

---

## 9️⃣ Sentiment Positivity (15%)
VADER compound mapped to score:

```
mapped = (compound + 1) / 2   # Convert -1..1 to 0..1
Match bands (<=0.3 → low, >0.8 → excellent)
```

---

## 🎯 Final Score Formula

```
overall = (sum(criteria_score * weight) / sum(weights)) * 100
