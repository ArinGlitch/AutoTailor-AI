from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Dict, Any
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Settings(BaseSettings):
    app_name: str = "AutoTailor API"
    class Config:
        env_file = ".env"

settings = Settings()
app = FastAPI(title=settings.app_name)

# --------- Schemas ---------
class TailorRequest(BaseModel):
    resume_text: str
    job_description: str
    variant: str | None = "standard"

class AnalyzeResponse(BaseModel):
    coverage: float
    fit_score: float
    missing_keywords: List[str]

class GenerateResponse(BaseModel):
    tailored_text: List[str]
    coverage: float
    fit_score: float
    missing_keywords: List[str]
    diffs: List[Dict[str, Any]]

# --------- Helpers ---------
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()

def top_keywords(text: str, k: int = 25) -> List[str]:
    # Simple TF-IDF over one doc is degenerate; so compare JD vs resume together
    # to get weighted tokens from the JD.
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    docs = [text, ""]  # prevent single-doc edge case
    X = vec.fit_transform(docs)
    # Build a quick TF view for JD tokens
    jd_vec = vec.transform([text])
    tfidf = jd_vec.toarray()[0]
    terms = vec.get_feature_names_out()
    ranked = sorted(zip(terms, tfidf), key=lambda t: t[1], reverse=True)
    return [w for w, score in ranked[:k] if w.isalpha() or " " in w]

def coverage_and_fit(jd: str, resume: str) -> tuple[float, float, List[str]]:
    # Compute TF-IDF vectors for fit; also compute coverage
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform([jd, resume])
    fit = float(cosine_similarity(X[0], X[1])[0][0])  # 0..1-ish

    jd_terms = set(top_keywords(jd, k=50))
    resume_terms = set(top_keywords(resume, k=200))
    overlap = jd_terms & resume_terms
    cov = (len(overlap) / max(1, len(jd_terms))) * 100.0
    missing = sorted(list(jd_terms - resume_terms))[:15]
    return cov, fit, missing

def extract_bullets(resume: str) -> List[str]:
    # Naive split by newlines or bullets. Improve later with spaCy/heuristics.
    lines = [l.strip("•- ").strip() for l in resume.splitlines() if l.strip()]
    # Keep only reasonably short "bullet-like" lines
    bullets = [l for l in lines if 5 <= len(l.split()) <= 40]
    return bullets[:20]

def select_bullets_for_jd(jd: str, bullets: List[str], top_n: int = 5) -> List[str]:
    if not bullets:
        return []
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform([jd] + bullets)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()  # similarity JD vs each bullet
    ranked_idx = sims.argsort()[::-1][:top_n]
    return [bullets[i] for i in ranked_idx]

def light_rewrite(bullet: str, jd_keywords: List[str]) -> str:
    # Minimal, deterministic “rewrite”: append up to 2 JD terms not already present.
    add = [k for k in jd_keywords if k.lower() not in bullet.lower()]
    add = add[:2]
    if not add:
        return bullet
    return f"{bullet} (aligned to: {', '.join(add)})"

# --------- Endpoints ---------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: TailorRequest):
    cov, fit, missing = coverage_and_fit(normalize(req.job_description), normalize(req.resume_text))
    return AnalyzeResponse(coverage=round(cov, 2), fit_score=round(fit, 3), missing_keywords=missing)

@app.post("/generate", response_model=GenerateResponse)
def generate(req: TailorRequest):
    jd = normalize(req.job_description)
    resume = normalize(req.resume_text)
    cov, fit, missing = coverage_and_fit(jd, resume)

    bullets = extract_bullets(req.resume_text)
    picked = select_bullets_for_jd(jd, bullets, top_n=5)

    # very light, non-fabricating rewrite using JD keywords
    jd_keys = top_keywords(jd, k=20)
    tailored = [light_rewrite(b, jd_keys) for b in picked]

    # Build “diffs” so a UI can show accept/reject
    diffs = [{"old": o, "new": n, "accepted": False} for o, n in zip(picked, tailored)]

    return GenerateResponse(
        tailored_text=tailored,
        coverage=round(cov, 2),
        fit_score=round(fit, 3),
        missing_keywords=missing,
        diffs=diffs
    )
