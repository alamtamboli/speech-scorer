# backend/app/score_engine.py
"""
Updated ScoreEngine with additional scoring features:
 - speech pacing variation (pacing_variation)
 - pause frequency (pause_frequency)
 - repetition / redundancy (repetition)
 - coherence & flow (coherence)

Integrates with existing rubric.json structure. Designed to degrade gracefully
if audio_stats or embedding model is unavailable.
"""
import json
import os
import re
from typing import Optional, List, Dict, Any

import statistics

from sentence_transformers import SentenceTransformer, util

# optional imports - gracefully degrade if not installed
try:
    import language_tool_python
except Exception:
    language_tool_python = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

from .neo4j_layer import Neo4jLayer


# -----------------------
# Helper functions
# -----------------------
def tokenize_words(text: str) -> List[str]:
    # simple tokenization (split on whitespace and punctuation)
    tokens = re.findall(r"\b\w+'?\w*\b", text.lower())
    return tokens


def compute_ttr(text: str) -> float:
    tokens = tokenize_words(text)
    if not tokens:
        return 0.0
    types = set(tokens)
    return len(types) / len(tokens)


def compute_wpm_from_audio(audio_stats: Optional[dict], transcript: str) -> float:
    # prefer audio-provided wpm
    if audio_stats and isinstance(audio_stats, dict) and audio_stats.get("wpm"):
        try:
            return float(audio_stats.get("wpm"))
        except Exception:
            pass
    # fallback: estimate speaking speed as 120 WPM (safe default)
    words = len(tokenize_words(transcript))
    if words == 0:
        return 0.0
    # If no audio, treat as ideal speaking speed (makes sense for text-only scoring)
    return 120.0


def compute_filler_rate(text: str, filler_list: List[str]) -> float:
    tokens = tokenize_words(text)
    if not tokens:
        return 0.0
    lc = text.lower()
    count = sum(lc.count(f) for f in filler_list)
    return count / max(1, len(tokens)) * 100.0  # convert to percent of words


def compute_grammar_error_rate(text: str) -> Optional[float]:
    if language_tool_python is None:
        return None
    try:
        tool = language_tool_python.LanguageTool("en-US")
        matches = tool.check(text)
        # language_tool returns many matches; we count major grammatical 'ERROR' types
        errors = 0
        for m in matches:
            # Consider only non-information/warning matches: use category or ruleId heuristics
            # language_tool messages have 'ruleId' and 'category'; we'll count all matches as errors
            errors += 1
        words = len(tokenize_words(text))
        if words == 0:
            return 0.0
        # errors per 100 words
        return errors / words * 100.0
    except Exception:
        return None


def compute_sentiment_score(text: str) -> Optional[float]:
    if SentimentIntensityAnalyzer is None:
        return None
    try:
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(text)
        # use compound score in range [-1,1] -> map to [0,1]
        return (vs.get("compound", 0.0) + 1.0) / 2.0
    except Exception:
        return None


def match_keyword_groups(transcript: str, groups: List[List[str]]) -> int:
    """Return number of groups matched (group matched if any keyword in group found)."""
    lc = transcript.lower()
    matched = 0
    for grp in groups:
        for kw in grp:
            if kw.lower() in lc:
                matched += 1
                break
    return matched


def check_order_flow(transcript: str, expected_order: List[str]) -> str:
    """
    Very simple order checking:
    - find index of first occurrence of each expected_order keyword (if present)
    - if majority of found indices are in ascending order -> 'Correct Order'
    - if some present but not ordered -> 'Partial Order'
    - if none present -> 'No Order'
    """
    lc = transcript.lower()
    indices = []
    for key in expected_order:
        # check a few likely tokens for each key
        if key == "greeting":
            terms = ["hello", "hi", "good morning", "good afternoon", "good evening"]
        elif key == "name":
            terms = ["my name is", "i am", "this is"]
        elif key == "education":
            terms = ["study", "studied", "degree", "college", "university", "school"]
        elif key == "family":
            terms = ["father", "mother", "brother", "sister", "family"]
        elif key == "experience":
            terms = ["worked", "experience", "job", "internship", "project"]
        elif key == "strengths":
            terms = ["strength", "skill", "skilled", "good at"]
        elif key == "weaknesses":
            terms = ["weakness", "improve", "learning", "struggle"]
        elif key == "hobbies":
            terms = ["hobby", "hobbies", "like to", "enjoy", "interest"]
        elif key == "goals":
            terms = ["goal", "aim", "future", "plan", "aspire"]
        else:
            terms = [key]

        found_index = None
        for t in terms:
            pos = lc.find(t)
            if pos != -1:
                if found_index is None or pos < found_index:
                    found_index = pos
        if found_index is not None:
            indices.append(found_index)

    if not indices:
        return "No Order"
    # check if indices in ascending order roughly
    if indices == sorted(indices):
        return "Correct Order"
    # partial: some in order but not all
    return "Partial Order"


# -----------------------
# NEW: Advanced audio/text metrics
# -----------------------
def compute_pacing_variation(audio_stats: Optional[dict]) -> Optional[float]:
    """
    Compute coefficient of variation (stdev/mean) of per-word durations.
    Expects audio_stats["word_timestamps"] = [{"start":..., "end":...}, ...]
    Returns coefficient (float) or None.
    """
    if not audio_stats or not isinstance(audio_stats, dict):
        return None
    word_ts = audio_stats.get("word_timestamps")
    if not word_ts or len(word_ts) < 3:
        return None
    durations = []
    for w in word_ts:
        try:
            start = float(w.get("start", 0.0))
            end = float(w.get("end", start))
            dur = max(0.0, end - start)
            durations.append(dur)
        except Exception:
            continue
    if len(durations) < 3:
        return None
    avg = statistics.mean(durations)
    stdev = statistics.stdev(durations)
    if avg == 0:
        return None
    cov = stdev / avg
    return round(cov, 4)


def compute_pause_frequency(audio_stats: Optional[dict]) -> Optional[int]:
    """
    Count number of long pauses (e.g., > 0.3s) present in audio_stats.
    Expects audio_stats["pauses"] = [0.2, 0.5, ...] durations in seconds
    Returns integer count or None.
    """
    if not audio_stats or not isinstance(audio_stats, dict):
        return None
    pauses = audio_stats.get("pauses")
    if not pauses or not isinstance(pauses, list):
        return None
    try:
        long_pauses = sum(1 for p in pauses if (p is not None and float(p) > 0.3))
        return int(long_pauses)
    except Exception:
        return None


def compute_repetition_score(transcript: str) -> int:
    """
    Simple repetition detector: counts exact repeated sentences.
    Returns integer count of repeated sentences (0 means no repetition).
    """
    if not transcript:
        return 0
    sentences = [s.strip() for s in re.split(r'[.!?]', transcript) if s.strip()]
    lower = [s.lower() for s in sentences]
    repeats = 0
    seen = {}
    for s in lower:
        seen[s] = seen.get(s, 0) + 1
    for cnt in seen.values():
        if cnt > 1:
            repeats += (cnt - 1)
    return repeats


def compute_coherence_score(embed_model: Optional[SentenceTransformer], transcript: str) -> float:
    """
    Compute average sentence-to-next-sentence semantic similarity mapped to [0,1].
    If embed_model is None or computation fails, return neutral 0.5.
    """
    if not transcript:
        return 0.5
    if embed_model is None:
        return 0.5
    sentences = [s.strip() for s in re.split(r'[.!?]', transcript) if s.strip()]
    if len(sentences) < 2:
        return 0.5
    try:
        # embeddings: list or tensor depending on model
        embeddings = embed_model.encode(sentences, convert_to_tensor=True)
        sims = []
        for i in range(len(sentences) - 1):
            try:
                sim = util.cos_sim(embeddings[i], embeddings[i + 1]).item()
                sims.append((sim + 1.0) / 2.0)  # map to [0,1]
            except Exception:
                continue
        if not sims:
            return 0.5
        avg = sum(sims) / len(sims)
        return round(float(avg), 4)
    except Exception:
        return 0.5


# -----------------------
# Main Scoring Engine
# -----------------------
class ScoreEngine:
    def __init__(self, rubric_path: Optional[str] = None):
        base = os.path.dirname(__file__)
        if not rubric_path:
            rubric_path = os.path.join(base, "rubric.json")
        with open(rubric_path, "r", encoding="utf-8") as f:
            self.rubric = json.load(f)

        # embeddings
        try:
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self.embed_model = None

        # neo4j layer
        self.neo = Neo4jLayer()

        # filler list default (from rubric)
        self.default_fillers = []
        # attempt to extract default filler list from rubric
        for cat in self.rubric.get("categories", []):
            for crit in cat.get("criteria", []):
                if crit.get("id") == "filler_rate":
                    self.default_fillers = crit.get("keywords", [])
        # fallback
        if not self.default_fillers:
            self.default_fillers = ["um", "uh", "like", "you know", "so", "actually"]

    def quick_rule_scan(self, transcript: str) -> Dict[str, Any]:
        t = transcript.strip()
        wc = len(tokenize_words(t))
        finds = {}
        for cat in self.rubric.get("categories", []):
            for crit in cat.get("criteria", []):
                kws = crit.get("keywords", [])
                found = [k for k in kws if k.lower() in t.lower()]
                finds[crit.get("id")] = {"found_keywords": found, "word_count": wc}
        return {"word_count": wc, "criteria": finds}

    def score_text(self, transcript: str, audio_stats: Optional[dict] = None) -> Dict[str, Any]:
        transcript = transcript.strip()
        if not transcript:
            return {"overall": 0.0, "per_criteria": []}

        words = tokenize_words(transcript)
        wc = len(words)

        # compute base signals
        ttr = compute_ttr(transcript)
        wpm = compute_wpm_from_audio(audio_stats, transcript)
        filler_rate_percent = compute_filler_rate(transcript, self.default_fillers)  # percent
        grammar_err_rate = compute_grammar_error_rate(transcript)  # errors per 100 words, or None
        sentiment_score = compute_sentiment_score(transcript)  # 0..1 or None

        # additional advanced signals
        pacing_cov = compute_pacing_variation(audio_stats)  # coefficient of variation or None
        pause_freq = compute_pause_frequency(audio_stats)  # integer or None
        repetition_count = compute_repetition_score(transcript)  # integer
        coherence = compute_coherence_score(self.embed_model, transcript)  # 0..1 float (or 0.5 default)

        # precompute transcript embedding if model available
        if self.embed_model:
            try:
                transcript_emb = self.embed_model.encode(transcript, convert_to_tensor=True)
            except Exception:
                transcript_emb = None
        else:
            transcript_emb = None

        per_results = []
        total_weight = 0.0
        weighted_sum = 0.0

        for cat in self.rubric.get("categories", []):
            crits = cat.get("criteria") or cat.get("criteria", [])
            for crit in crits:
                cid = crit.get("id")
                name = crit.get("name")
                weight = crit.get("weight", 0)
                score_components = {}
                crit_score = 0.0
                band_label = None

                # ---------- existing criteria (kept as before) ----------
                if cid == "situation_level":
                    bands = crit.get("bands", [])
                    found_score = 0
                    for b in bands:
                        kws = b.get("keywords", [])
                        matched = any(k.lower() in transcript.lower() for k in kws)
                        if matched:
                            found_score = max(found_score, b.get("score", 0))
                            band_label = b.get("label")
                    if found_score == 0:
                        if "introduce" in transcript.lower() or "my name is" in transcript.lower():
                            found_score = next((b.get("score", 0) for b in bands if b.get("label", "").lower() == "good"), 0)
                            band_label = "Good"
                    maxband = max([b.get("score", 0) for b in bands]) if bands else 1
                    crit_score = (found_score / max(1, maxband)) if maxband > 0 else 0.0
                    score_components["band_raw"] = found_score
                    score_components["band_max"] = maxband

                elif cid == "keyword_relevance":
                    groups = crit.get("keyword_groups", [])
                    matched_groups = match_keyword_groups(transcript, groups)
                    band_score = 0
                    band_label = "Poor"
                    for b in crit.get("bands", []):
                        min_groups = b.get("min_groups", 0)
                        if matched_groups >= min_groups:
                            band_score = b.get("score", 0)
                            band_label = b.get("label")
                            break
                    best = None
                    for b in crit.get("bands", []):
                        if matched_groups >= b.get("min_groups", 0):
                            if best is None or b.get("min_groups", 0) > best.get("min_groups", 0):
                                best = b
                    if best:
                        band_score = best.get("score", 0)
                        band_label = best.get("label")
                    maxband = max((b.get("score", 0) for b in crit.get("bands", [])), default=1)
                    crit_score = (band_score / max(1, maxband)) if maxband > 0 else 0.0
                    score_components["matched_groups"] = matched_groups
                    score_components["band_raw"] = band_score
                    score_components["band_max"] = maxband

                elif cid == "order_flow":
                    expected_order = crit.get("expected_order", [])
                    order_result = check_order_flow(transcript, expected_order)
                    if order_result == "Correct Order":
                        raw = next((b.get("score", 0) for b in crit.get("bands", []) if b.get("label") == "Correct Order"), 0)
                        band_label = "Correct Order"
                    elif order_result == "Partial Order":
                        raw = next((b.get("score", 0) for b in crit.get("bands", []) if b.get("label") == "Partial Order"), 0)
                        band_label = "Partial Order"
                    else:
                        raw = next((b.get("score", 0) for b in crit.get("bands", []) if b.get("label") == "No Order"), 0)
                        band_label = "No Order"
                    maxband = max((b.get("score", 0) for b in crit.get("bands", [])), default=1)
                    crit_score = (raw / max(1, maxband)) if maxband > 0 else 0.0
                    score_components["order_result"] = order_result
                    score_components["band_raw"] = raw
                    score_components["band_max"] = maxband

                elif cid == "length":
                    min_w = crit.get("min_words")
                    max_w = crit.get("max_words")
                    if min_w and wc < min_w:
                        val = wc / min_w
                    elif max_w and wc > max_w:
                        val = max(0.0, 1.0 - (wc - max_w) / max_w)
                    else:
                        val = 1.0
                    crit_score = max(0.0, min(1.0, val))
                    score_components["word_count"] = wc

                # SPEECH RATE
                elif cid == "wpm_rate":
                    bands = crit.get("bands", [])
                    chosen_score = 0
                    chosen_label = None
                    for b in bands:
                        r = b.get("range", "")
                        s = r.strip()
                        score_val = b.get("score", 0)
                        match = False
                        if s.endswith("+"):
                            try:
                                low = int(s[:-1])
                                if wpm >= low:
                                    match = True
                            except:
                                pass
                        elif s.startswith("<"):
                            try:
                                hi = int(s[1:])
                                if wpm < hi:
                                    match = True
                            except:
                                pass
                        elif "-" in s:
                            try:
                                lo, hi = s.split("-", 1)
                                lo = int(lo); hi = int(hi)
                                if wpm >= lo and wpm <= hi:
                                    match = True
                            except:
                                pass
                        if match:
                            chosen_score = score_val
                            chosen_label = b.get("label")
                            break
                    maxband = max((b.get("score", 0) for b in bands), default=1)
                    crit_score = (chosen_score / max(1, maxband)) if maxband > 0 else 0.0
                    band_label = chosen_label
                    score_components["wpm"] = round(wpm, 2)
                    score_components["band_raw"] = chosen_score

                # LANGUAGE & GRAMMAR
                elif cid == "grammar_errors":
                    raw_rate = grammar_err_rate
                    chosen_score = 0
                    chosen_label = None
                    if raw_rate is None:
                        raw_rate = 0.0
                    for b in crit.get("bands", []):
                        rng = b.get("range", "")
                        if "+" in rng:
                            try:
                                lo = float(rng.replace("+", ""))
                                if raw_rate >= lo:
                                    chosen_score = b.get("score", 0)
                                    chosen_label = rng
                                    break
                            except:
                                pass
                        elif "-" in rng:
                            try:
                                lo, hi = rng.split("-", 1)
                                lo = float(lo); hi = float(hi)
                                if raw_rate >= lo and raw_rate <= hi:
                                    chosen_score = b.get("score", 0)
                                    chosen_label = rng
                                    break
                            except:
                                pass
                    maxband = max((b.get("score", 0) for b in crit.get("bands", [])), default=1)
                    crit_score = (chosen_score / max(1, maxband)) if maxband > 0 else 0.0
                    band_label = chosen_label
                    score_components["grammar_errors_per100"] = round(raw_rate, 3)
                    score_components["band_raw"] = chosen_score

                elif cid == "vocab":
                    raw_ttr = ttr
                    chosen_score = 0
                    chosen_label = None
                    for b in crit.get("bands", []):
                        rng = b.get("range", "")
                        if "-" in rng:
                            try:
                                lo, hi = rng.split("-", 1)
                                lo = float(lo); hi = float(hi)
                                if raw_ttr >= lo and raw_ttr <= hi:
                                    chosen_score = b.get("score", 0)
                                    chosen_label = rng
                                    break
                            except:
                                pass
                    maxband = max((b.get("score", 0) for b in crit.get("bands", [])), default=1)
                    crit_score = (chosen_score / max(1, maxband)) if maxband > 0 else 0.0
                    score_components["ttr"] = round(raw_ttr, 3)
                    score_components["band_raw"] = chosen_score

                # CLARITY -> filler_rate
                elif cid == "filler_rate":
                    raw_percent = filler_rate_percent  # percent of words
                    chosen_score = 0
                    chosen_label = None
                    for b in crit.get("bands", []):
                        rng = b.get("range", "")
                        if "+" in rng:
                            try:
                                lo = float(rng.replace("+", ""))
                                if raw_percent >= lo:
                                    chosen_score = b.get("score", 0)
                                    chosen_label = rng
                                    break
                            except:
                                pass
                        elif "-" in rng:
                            try:
                                lo, hi = rng.split("-", 1)
                                lo = float(lo); hi = float(hi)
                                if raw_percent >= lo and raw_percent <= hi:
                                    chosen_score = b.get("score", 0)
                                    chosen_label = rng
                                    break
                            except:
                                pass
                    maxband = max((b.get("score", 0) for b in crit.get("bands", [])), default=1)
                    crit_score = (chosen_score / max(1, maxband)) if maxband > 0 else 0.0
                    band_label = chosen_label
                    score_components["filler_percent"] = round(raw_percent, 3)
                    score_components["band_raw"] = chosen_score

                # ENGAGEMENT -> sentiment
                elif cid == "sentiment":
                    raw_sent = sentiment_score
                    chosen_score = 0
                    chosen_label = None
                    if raw_sent is None:
                        raw_sent = 0.5
                    for b in crit.get("bands", []):
                        rng = b.get("range", "")
                        if rng.startswith(">"):
                            try:
                                lo = float(rng.replace(">", ""))
                                if raw_sent > lo:
                                    chosen_score = b.get("score", 0)
                                    chosen_label = rng
                                    break
                            except:
                                pass
                        elif "<=" in rng:
                            try:
                                hi = float(rng.replace("<=", ""))
                                if raw_sent <= hi:
                                    chosen_score = b.get("score", 0)
                                    chosen_label = rng
                                    break
                            except:
                                pass
                        elif "-" in rng:
                            try:
                                lo, hi = rng.split("-", 1)
                                lo = float(lo); hi = float(hi)
                                if raw_sent >= lo and raw_sent <= hi:
                                    chosen_score = b.get("score", 0)
                                    chosen_label = rng
                                    break
                            except:
                                pass
                    maxband = max((b.get("score", 0) for b in crit.get("bands", [])), default=1)
                    crit_score = (chosen_score / max(1, maxband)) if maxband > 0 else 0.0
                    band_label = chosen_label
                    score_components["sentiment_raw"] = round(raw_sent, 3)
                    score_components["band_raw"] = chosen_score

                # ---------- NEW CRITERIA ----------
                elif cid == "pacing_variation":
                    # lower coefficient of variation (cov) is better.
                    # mapping: cov <=0.10 -> full; <=0.20 -> 0.85; <=0.40 -> 0.6; <=0.6 -> 0.4; >0.6 -> 0.2
                    cov = pacing_cov
                    score_components["pacing_cov"] = cov
                    if cov is None:
                        # neutral fallback score
                        crit_score = 0.5
                    else:
                        if cov <= 0.10:
                            crit_score = 1.0
                            band_label = "<=0.10"
                        elif cov <= 0.20:
                            crit_score = 0.85
                            band_label = "0.11-0.20"
                        elif cov <= 0.40:
                            crit_score = 0.6
                            band_label = "0.21-0.40"
                        elif cov <= 0.60:
                            crit_score = 0.4
                            band_label = "0.41-0.60"
                        else:
                            crit_score = 0.2
                            band_label = ">0.60"
                    score_components["band_raw"] = round(cov, 4) if cov is not None else None

                elif cid == "pause_frequency":
                    # fewer long pauses is better. pause_freq = count of pauses > 0.3s
                    pf = pause_freq
                    score_components["pause_count"] = pf
                    if pf is None:
                        crit_score = 0.5
                    else:
                        if pf == 0:
                            crit_score = 1.0
                            band_label = "0"
                        elif pf == 1:
                            crit_score = 0.9
                            band_label = "1"
                        elif pf == 2:
                            crit_score = 0.7
                            band_label = "2"
                        elif pf == 3:
                            crit_score = 0.5
                            band_label = "3"
                        else:
                            crit_score = 0.2
                            band_label = "4+"
                    score_components["band_raw"] = pf

                elif cid == "repetition":
                    # fewer repetitions -> better. repetition_count is integer
                    rep = repetition_count
                    score_components["repetition_count"] = rep
                    if rep <= 0:
                        crit_score = 1.0
                        band_label = "0"
                    elif rep == 1:
                        crit_score = 0.8
                        band_label = "1"
                    elif rep == 2:
                        crit_score = 0.6
                        band_label = "2"
                    elif rep == 3:
                        crit_score = 0.4
                        band_label = "3"
                    else:
                        crit_score = 0.2
                        band_label = "4+"
                    score_components["band_raw"] = rep

                elif cid == "coherence":
                    # coherence is already 0..1 from compute_coherence_score
                    coh = coherence
                    score_components["coherence_raw"] = coh
                    if coh is None:
                        crit_score = 0.5
                    else:
                        # map directly
                        crit_score = max(0.0, min(1.0, float(coh)))
                        if crit_score >= 0.85:
                            band_label = "High"
                        elif crit_score >= 0.65:
                            band_label = "Good"
                        elif crit_score >= 0.45:
                            band_label = "Average"
                        else:
                            band_label = "Low"
                    score_components["band_raw"] = round(coh, 4) if coh is not None else None

                # ---------- fallback generic semantic/keyword/length scoring ----------
                else:
                    sem_score = 0.0
                    desc = crit.get("description", crit.get("name", ""))
                    if transcript_emb is not None and desc and self.embed_model:
                        try:
                            desc_emb = self.embed_model.encode(desc, convert_to_tensor=True)
                            sim = util.cos_sim(transcript_emb, desc_emb).item()
                            sem_score = max(0.0, min(1.0, (sim + 1.0) / 2.0))
                        except Exception:
                            sem_score = 0.0
                    kws = crit.get("keywords", [])
                    if kws:
                        kw_matches = sum(1 for k in kws if k.lower() in transcript.lower())
                        kw_score = kw_matches / max(1, len(kws))
                    else:
                        kw_score = 1.0
                        kw_matches = 0
                    length_score = 1.0
                    if crit.get("min_words") or crit.get("max_words"):
                        min_w = crit.get("min_words")
                        max_w = crit.get("max_words")
                        if min_w and wc < min_w:
                            length_score = wc / min_w
                        elif max_w and wc > max_w:
                            length_score = max(0.0, 1.0 - (wc - max_w) / max_w)
                    crit_raw = 0.6 * sem_score + 0.3 * kw_score + 0.1 * length_score
                    try:
                        graph_score = self.neo.criterion_relevance(cid, transcript)
                    except Exception:
                        graph_score = 0.0
                    combined = 0.85 * crit_raw + 0.15 * graph_score
                    crit_score = max(0.0, min(1.0, combined))
                    score_components["sem_score"] = round(sem_score, 4)
                    score_components["kw_matches"] = kw_matches
                    score_components["length_score"] = round(length_score, 4)
                    score_components["graph_score"] = round(graph_score, 4)

                # append criterion result
                per_results.append({
                    "category": cat.get("id"),
                    "id": cid,
                    "name": name,
                    "weight": weight,
                    "score": round(float(crit_score), 4),
                    "band": band_label,
                    "components": score_components
                })

                total_weight += weight
                weighted_sum += crit_score * weight

        overall = (weighted_sum / max(total_weight, 1)) * 100.0
        return {"overall": round(overall, 2), "per_criteria": per_results}

    def close(self):
        try:
            self.neo.close()
        except Exception:
            pass
