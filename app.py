"""
AI Candidate Screening Platform
================================
A Streamlit application that automates candidate screening using Google's Gemini AI.
Designed with a minimal, professional B2B SaaS aesthetic.
"""
import os
from dotenv import load_dotenv
load_dotenv(override=True)
import io
import json
import re
import time
import datetime
import html
from typing import Optional

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

st.set_page_config(
    page_title="AI Recruitment Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling & UI Components
# ---------------------------------------------------------------------------

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --app-font: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            --app-border: color-mix(in srgb, var(--text-color) 14%, transparent);
            --app-muted: color-mix(in srgb, var(--text-color) 62%, transparent);
            --app-subtle: color-mix(in srgb, var(--text-color) 44%, transparent);
            --app-card: var(--secondary-background-color);
            --app-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
            --app-radius: 8px;
            --success-text: #15803D;
            --warning-text: #B45309;
            --danger-text: #B91C1C;
            --success-bg: rgba(34, 197, 94, 0.12);
            --warning-bg: rgba(245, 158, 11, 0.12);
            --danger-bg: rgba(239, 68, 68, 0.12);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --success-text: #86EFAC;
                --warning-text: #FCD34D;
                --danger-text: #FCA5A5;
                --success-bg: rgba(34, 197, 94, 0.18);
                --warning-bg: rgba(245, 158, 11, 0.18);
                --danger-bg: rgba(239, 68, 68, 0.18);
            }
        }
        
        /* 1. Global Visibility */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* 2. Global Typography (Safe Approach) */
        /* Applies Inter font to everything without breaking layout or colors */
        html, body, .stApp {
            font-family: var(--app-font);
            -webkit-font-smoothing: antialiased;
        }
        
        /* 3. App Layout */
        .block-container {
            padding-top: 80px;
            padding-bottom: 64px;
            max-width: 1120px;
        }
        
        /* 4. Custom UI Components: Typography */
        /* Use inherit for color so Dark/Light modes work seamlessly */
        .page-title { font-size: 26px; font-weight: 600; margin-bottom: 8px; }
        .page-subtitle { font-size: 15px; color: var(--app-muted); margin-bottom: 40px; font-weight: 400; line-height: 1.5; }
        .section-title { font-size: 14px; font-weight: 600; margin-bottom: 16px; margin-top: 32px; }

        /* 5. Custom UI Components: Metric Cards */
        .metric-card {
            background-color: var(--app-card);
            border: 1px solid var(--app-border);
            border-radius: var(--app-radius);
            padding: 24px;
            box-shadow: var(--app-shadow);
            display: flex;
            flex-direction: column;
            gap: 4px;
            height: 100%;
        }
        .metric-label { font-size: 13px; font-weight: 500; color: var(--app-muted); }
        .metric-value { font-size: 36px; font-weight: 600; line-height: 1.1; margin: 4px 0; }
        .metric-caption { font-size: 13px; font-weight: 400; color: var(--app-subtle); margin-top: 2px; }
        
        /* 6. Custom UI Components: Action Cards */
        .action-card {
        transition: transform 0.15s ease, box-shadow 0.15s ease;
            background-color: var(--app-card);
            border: 1px solid var(--app-border);
            border-radius: var(--app-radius);
            padding: 24px;
            box-shadow: var(--app-shadow);
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        .action-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,.08);
}
        .action-title { font-size: 15px; font-weight: 600; margin-bottom: 8px; }
        .action-desc { font-size: 14px; color: var(--app-muted); margin-bottom: 24px; flex-grow: 1; line-height: 1.5; }
        
        /* 7. Custom UI Components: Candidate Cards */
        .candidate-card {
        transition: transform 0.15s ease, box-shadow 0.15s ease;
            background-color: var(--app-card);
            border: 1px solid var(--app-border);
            border-radius: var(--app-radius);
            padding: 32px;
            margin-bottom: 16px;
            box-shadow: var(--app-shadow);
        }
        .candidate-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,.08);
}
        .cc-divider {
            height: 1px;
            background-color: var(--app-border);
            margin: 24px 0;
            width: 100%;
        }
        .cc-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
        .cc-title { font-size: 18px; font-weight: 600; margin-bottom: 4px; }
        .cc-subtitle { font-size: 14px; color: var(--app-muted); }
        .cc-badges { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
        .cc-summary { font-size: 14px; line-height: 1.6; max-width: 85%; }
        .cc-details { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
        .cc-section-title { font-size: 12px; font-weight: 600; color: var(--app-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; }
        .cc-list { margin: 0; padding: 0; font-size: 14px; list-style-type: none; line-height: 1.5; }
        .cc-list li { margin-bottom: 8px; padding-left: 12px; position: relative; }
        .cc-list li::before { content: "•"; position: absolute; left: 0; color: var(--app-subtle); }
        
        .xai-panel { margin-top: 20px; max-width: 85%; }
        .xai-row { display: grid; grid-template-columns: 96px 1fr; gap: 16px; align-items: center; margin-bottom: 10px; }
        .xai-label { font-size: 13px; color: var(--app-muted); }
        .xai-track { width: 100%; height: 8px; background-color: var(--app-border); border-radius: 6px; overflow: hidden; }
        .xai-fill { height: 100%; background-color: var(--primary-color); border-radius: 6px; }
        .xai-confidence { display: flex; gap: 8px; align-items: center; margin-top: 14px; font-size: 13px; color: var(--app-muted); }
        .xai-confidence strong { color: var(--text-color); font-weight: 600; }
        .xai-reasoning { margin-top: 8px; font-size: 13px; line-height: 1.5; color: var(--app-muted); }

        /* 8. Custom UI Components: Progress Bar */
        .cc-progress-container { width: 100%; background-color: var(--app-border); border-radius: 6px; height: 8px; overflow: hidden; }
        .cc-progress-bar { height: 100%; border-radius: 6px; }
        
        /* 9. Custom UI Components: Badges & Colors */
        .badge { padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 500; border: 1px solid transparent; }
        .badge-success { background-color: var(--success-bg); color: var(--success-text); border-color: color-mix(in srgb, var(--success-text) 28%, transparent); }
        .badge-warning { background-color: var(--warning-bg); color: var(--warning-text); border-color: color-mix(in srgb, var(--warning-text) 28%, transparent); }
        .badge-danger { background-color: var(--danger-bg); color: var(--danger-text); border-color: color-mix(in srgb, var(--danger-text) 28%, transparent); }
        .badge-neutral { background-color: transparent; border-color: var(--app-border); color: var(--text-color); }
        .bg-success { background-color: #22C55E; }
        .bg-warning { background-color: #F59E0B; }
        .bg-danger { background-color: #EF4444; }
        
        /* 10. Custom UI Components: Ready Check Panel */
        .ready-check-panel {
            background-color: var(--app-card);
            border: 1px solid var(--app-border);
            border-radius: var(--app-radius);
            padding: 24px;
            margin-bottom: 32px;
            box-shadow: var(--app-shadow);
        }
        .ready-check-item { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; font-size: 14px; }
        .ready-check-item:last-child { margin-bottom: 0; }
        .indicator { width: 8px; min-width: 8px; height: 8px; border-radius: 50%; box-shadow: 0 0 0 2px var(--app-card); }
        .ind-green { background-color: #22C55E; outline: 1px solid #22C55E; outline-offset: 1px; }
        .ind-red { background-color: #EF4444; }
        .ind-gray { background-color: var(--app-subtle); }
        
        /* 11. Custom UI Components: Empty States & Tables */
        .empty-state {
            background-color: transparent;
            border: 1px dashed var(--app-border);
            border-radius: var(--app-radius);
            padding: 48px 24px;
            text-align: center;
            color: var(--app-muted);
            font-size: 14px;
            margin-top: 16px;
        }
        .table-container {
            border: 1px solid var(--app-border);
            border-radius: var(--app-radius);
            padding: 16px;
            background: var(--app-card);
            box-shadow: var(--app-shadow);
            margin-bottom: 16px;
        }

        /* 12. Streamlit Target Overrides (Safe) */
        
        /* Target Streamlit widgets with stable component test ids only */
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
            border-radius: 8px;
        }
        div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--primary-color);
}
        
        /* Target inputs safely */
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea {
            border-radius: 8px;
        }

        div[data-testid="stFileUploader"] {
            margin-top: 0;
            margin-bottom: 16px;
        }
        div[data-testid="stDataFrame"] {
            overflow: hidden;
            border-radius: 6px;
        }
        
        /* Sidebar styling safely */
        section[data-testid="stSidebar"] .block-container {
            padding-top: 32px;
        }
        
        div[data-testid="stPills"] button {
            border-radius: 6px;
            margin: 0;
            justify-content: flex-start;
        }
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 32px;
        }
        .sidebar-logo {
            width: 16px;
            min-width: 16px;
            height: 16px;
            background-color: var(--primary-color);
            border-radius: 4px;
        }
        .sidebar-title {
            font-weight: 600;
            font-size: 14px;
        }
        .sidebar-subtitle,
        .sidebar-section,
        .sidebar-status,
        .sidebar-row,
        .upload-note,
        .success-note {
            color: var(--app-muted);
        }
        .sidebar-subtitle { font-size: 12px; }
        .sidebar-section {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 12px;
        }
        .sidebar-status,
        .sidebar-row,
        .upload-note,
        .success-note {
            font-size: 13px;
        }
        .sidebar-api-state {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 12px;
            margin-bottom: 32px;
        }
        .sidebar-row {
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            gap: 12px;
        }
        .success-note {
            color: var(--success-text);
            margin-bottom: 16px;
        }
        .toolbar-spacer {
            min-height: 28px;
        }

        @media (max-width: 760px) {
            .page-subtitle { margin-bottom: 28px; }
            .candidate-card { padding: 24px; }
            .cc-header,
            .cc-details {
                display: block;
            }
            .cc-badges {
                justify-content: flex-start;
                margin-top: 12px;
            }
            .cc-summary {
                max-width: 100%;
            }
            .xai-panel {
                max-width: 100%;
            }
            .xai-row {
                grid-template-columns: 1fr;
                gap: 6px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def html_metric_card(label: str, value: str, caption: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-caption">{caption}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Helper — Google Drive URL conversion
# ---------------------------------------------------------------------------

def convert_gdrive_url(url: str) -> Optional[str]:
    if not url or not isinstance(url, str): return None
    url = url.strip()
    if "drive.google.com/uc?" in url and "export=download" in url: return url
    match = re.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", url)
    if match: return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    match = re.search(r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)", url)
    if match: return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    match = re.search(r"docs\.google\.com/\w+/d/([a-zA-Z0-9_-]+)", url)
    if match: return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    return url


# ---------------------------------------------------------------------------
# Helper — Download file from URL
# ---------------------------------------------------------------------------

def download_file(url: str, timeout: int = 30) -> Optional[bytes]:
    try:
        download_url = convert_gdrive_url(url)
        if not download_url: return None
        session = requests.Session()
        response = session.get(download_url, timeout=timeout, stream=True)
        response.raise_for_status()
        if "drive.google.com" in download_url:
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    download_url = f"{download_url}&confirm={value}"
                    response = session.get(download_url, timeout=timeout, stream=True)
                    response.raise_for_status()
                    break
        content = response.content
        if len(content) < 100: return None
        return content
    except requests.RequestException:
        return None


# ---------------------------------------------------------------------------
# Helper — Extract text from PDF bytes
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text_parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts).strip()
    except Exception as exc:
        return f"[PDF extraction error: {exc}]"


# ---------------------------------------------------------------------------
# Helper — Gemini AI evaluation
# ---------------------------------------------------------------------------

def evaluate_candidate_with_gemini(candidate_info: dict, resume_text: str, job_description: str, api_key: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    candidate_context = "\n".join(f"- {k}: {v}" for k, v in candidate_info.items() if v and str(v).strip())

    prompt = f"""You are an expert technical recruiter. Evaluate the following candidate against the provided job description. Be fair, objective, and concise.
=== JOB DESCRIPTION ===\n{job_description}\n=== CANDIDATE INFORMATION ===\n{candidate_context}\n=== RESUME TEXT ===\n{resume_text if resume_text else "[No resume text available]"}
=== INSTRUCTIONS ===
Return your evaluation as valid JSON with EXACTLY these keys:
{{"score": <integer 1-100>, "recommendation": "<Shortlist | Maybe | Reject>", "strengths": ["s1", "s2", "s3"], "weaknesses": ["w1", "w2"], "summary": "<2-3 sentence assessment>"}}
Score guide: 80-100: Shortlist, 50-79: Maybe, 1-49: Reject. Return ONLY the JSON object."""

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            
        result = json.loads(raw)
        required = {"score", "recommendation", "strengths", "weaknesses", "summary"}
        if not required.issubset(result.keys()): raise ValueError("Missing keys")
        
        result["score"] = max(1, min(100, int(result["score"])))
        rec = str(result.get("recommendation", "")).strip().lower()
        if "shortlist" in rec or "strong" in rec: result["recommendation"] = "Shortlist"
        elif "maybe" in rec or "partial" in rec or "consider" in rec: result["recommendation"] = "Maybe"
        else: result["recommendation"] = "Reject"
        return result
    except json.JSONDecodeError:
        return {"score": 0, "recommendation": "Reject", "strengths": ["None"], "weaknesses": ["Parsing Error"], "summary": "Failed to parse AI response."}
    except Exception as exc:
        return {"score": 0, "recommendation": "Reject", "strengths": ["None"], "weaknesses": ["API Error"], "summary": f"Gemini API error: {str(exc)}"}


# ---------------------------------------------------------------------------
# Helpers — Columns & Formatting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Optional Explainable AI extension
# ---------------------------------------------------------------------------

def generate_ai_explanation_with_gemini(candidate_info: dict, resume_text: str, job_description: str, evaluation: dict, api_key: str) -> Optional[dict]:
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        candidate_context = "\n".join(f"- {k}: {v}" for k, v in candidate_info.items() if v and str(v).strip())

        prompt = f"""You are an explainable AI assistant for recruiter screening.
Generate an explanation only for the already completed evaluation. Do not change the score, recommendation, summary, strengths, or weaknesses.

=== JOB DESCRIPTION ===
{job_description}

=== CANDIDATE INFORMATION ===
{candidate_context}

=== RESUME TEXT ===
{resume_text if resume_text else "[No resume text available]"}

=== EXISTING EVALUATION JSON ===
{json.dumps(evaluation)}

Return ONLY valid JSON with EXACTLY this structure:
{{"skills": <integer 0-100>, "projects": <integer 0-100>, "education": <integer 0-100>, "experience": <integer 0-100>, "overall_reasoning": "<brief explanation>", "confidence": "<High | Medium | Low>"}}"""

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        explanation = json.loads(raw)
        required = {"skills", "projects", "education", "experience", "overall_reasoning", "confidence"}
        if not required.issubset(explanation.keys()):
            return None

        for key in ["skills", "projects", "education", "experience"]:
            explanation[key] = max(0, min(100, int(explanation.get(key, 0))))

        confidence = str(explanation.get("confidence", "")).strip().title()
        explanation["confidence"] = confidence if confidence in {"High", "Medium", "Low"} else "Medium"
        explanation["overall_reasoning"] = str(explanation.get("overall_reasoning", "")).strip()
        return explanation
    except Exception:
        return None


def html_ai_explanation_breakdown(explanation: Optional[dict]) -> str:
    if not explanation:
        return ""

    try:
        rows = [
            ("Skills", int(explanation.get("skills", 0))),
            ("Projects", int(explanation.get("projects", 0))),
            ("Education", int(explanation.get("education", 0))),
            ("Experience", int(explanation.get("experience", 0))),
        ]
        bars = "".join(
            f"""
            <div class="xai-row">
                <div class="xai-label">{label}</div>
                <div class="xai-track"><div class="xai-fill" style="width: {max(0, min(100, value))}%;"></div></div>
            </div>
            """
            for label, value in rows
        )
        confidence = html.escape(str(explanation.get("confidence", "Medium")))
        reasoning = html.escape(str(explanation.get("overall_reasoning", "")))
    except Exception:
        return ""

    return f"""
    <div class="xai-panel">
        <div class="cc-section-title">AI Score Breakdown</div>
        {bars}
        <div class="xai-confidence"><span>Confidence</span><strong>{confidence}</strong></div>
        <div class="xai-reasoning">{reasoning}</div>
    </div>
    """


RESUME_COLUMN_NAMES = ["resume", "resume_link", "resume_url", "cv", "cv_link", "cv_url", "resume link", "resume url"]
NAME_COLUMN_NAMES = ["name", "candidate_name", "full_name", "candidate name", "full name", "applicant"]

def find_resume_column(df: pd.DataFrame) -> Optional[str]:
    lower_map = {col.lower().strip(): col for col in df.columns}
    for c in RESUME_COLUMN_NAMES:
        if c in lower_map: return lower_map[c]
    return None

def find_name_column(df: pd.DataFrame) -> Optional[str]:
    lower_map = {col.lower().strip(): col for col in df.columns}
    for c in NAME_COLUMN_NAMES:
        if c in lower_map: return lower_map[c]
    return df.columns[0] if len(df.columns) > 0 else None

def get_score_css(score: int) -> str:
    if score >= 80: return "badge-success", "bg-success"
    if score >= 50: return "badge-warning", "bg-warning"
    return "badge-danger", "bg-danger"

def get_status_css(rec: str) -> str:
    mapping = {"Shortlist": "badge-success", "Maybe": "badge-warning", "Reject": "badge-danger"}
    return mapping.get(normalize_recommendation(rec), "badge-neutral")

def normalize_recommendation(label: str) -> str:
    """Map any recommendation variant to the canonical form: Shortlist, Maybe, or Reject."""
    if not label or not isinstance(label, str):
        return "Reject"
    norm = label.strip().lower()
    if "shortlist" in norm or "strong" in norm:
        return "Shortlist"
    if "maybe" in norm or "consider" in norm or "partial" in norm:
        return "Maybe"
    return "Reject"


# ---------------------------------------------------------------------------
# GitHub Intelligence Module
# ---------------------------------------------------------------------------

GITHUB_COLUMN_NAMES = ["github", "github profile", "github url", "github_profile", "github_url", "github link"]

def find_github_column(df: pd.DataFrame) -> Optional[str]:
    lower_map = {col.lower().strip(): col for col in df.columns}
    for c in GITHUB_COLUMN_NAMES:
        if c in lower_map: return lower_map[c]
    return None

def extract_github_username(value: str) -> Optional[str]:
    if not value or not isinstance(value, str): return None
    value = value.strip()
    if not value: return None
    value = re.sub(r"^@", "", value)
    match = re.search(r"github\.com/([^/?#\s]+)", value, re.IGNORECASE)
    username = match.group(1) if match else value.rstrip("/").split("/")[-1]
    username = username.strip()
    if not re.match(r"^[A-Za-z0-9-]+$", username): return None
    return username

def github_api_headers() -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers

@st.cache_data(show_spinner=False)
def cache_github_profile(username: str) -> dict:
    try:
        response = requests.get(f"https://api.github.com/users/{username}", headers=github_api_headers(), timeout=20)
        if response.status_code in {403, 404, 410}:
            return {"ok": False, "error": "GitHub analysis unavailable."}
        response.raise_for_status()
        data = response.json()
        return {
            "ok": True,
            "profile": {
                "username": data.get("login", username),
                "name": data.get("name") or "",
                "bio": data.get("bio") or "",
                "followers": data.get("followers", 0),
                "following": data.get("following", 0),
                "public_repositories": data.get("public_repos", 0),
                "created_at": data.get("created_at") or "",
                "updated_at": data.get("updated_at") or "",
                "html_url": data.get("html_url") or f"https://github.com/{username}",
            },
        }
    except Exception:
        return {"ok": False, "error": "GitHub analysis unavailable."}

@st.cache_data(show_spinner=False)
def fetch_github_repositories(username: str) -> dict:
    try:
        params = {"sort": "updated", "direction": "desc", "per_page": 10}
        response = requests.get(f"https://api.github.com/users/{username}/repos", headers=github_api_headers(), params=params, timeout=20)
        if response.status_code in {403, 404, 410}:
            return {"ok": False, "error": "GitHub analysis unavailable.", "repositories": []}
        response.raise_for_status()
        repositories = []
        for repo in response.json():
            repositories.append({
                "name": repo.get("name") or "",
                "description": repo.get("description") or "",
                "language": repo.get("language") or "Unknown",
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "updated_at": repo.get("updated_at") or "",
                "size": repo.get("size", 0),
                "html_url": repo.get("html_url") or "",
            })
        return {"ok": True, "repositories": repositories}
    except Exception:
        return {"ok": False, "error": "GitHub analysis unavailable.", "repositories": []}

def fetch_github_profile(username: str) -> dict:
    return cache_github_profile(username)

def summarize_github_languages(repositories: list[dict]) -> dict:
    try:
        languages = {}
        for repo in repositories:
            language = repo.get("language") or "Unknown"
            languages[language] = languages.get(language, 0) + 1
        return dict(sorted(languages.items(), key=lambda item: item[1], reverse=True))
    except Exception:
        return {}

def generate_github_ai_analysis(profile: dict, repositories: list[dict], api_key: str) -> Optional[dict]:
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        prompt = f"""You are a senior technical recruiter evaluating only a candidate's GitHub profile.
Do not consider resumes, candidate ranking, or previous AI evaluation scores.

=== GITHUB PROFILE ===
{json.dumps(profile)}

=== LATEST REPOSITORIES ===
{json.dumps(repositories)}

Return ONLY valid JSON with exactly this structure:
{{"github_score": 87, "technical_strengths": ["...", "...", "..."], "activity_level": "High", "language_summary": "...", "open_source_maturity": "...", "summary": "...", "confidence": "High"}}"""

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        analysis = json.loads(raw)
        required = {"github_score", "technical_strengths", "activity_level", "language_summary", "open_source_maturity", "summary", "confidence"}
        if not required.issubset(analysis.keys()):
            return None
        analysis["github_score"] = max(0, min(100, int(analysis.get("github_score", 0))))
        if not isinstance(analysis.get("technical_strengths"), list):
            analysis["technical_strengths"] = []
        return analysis
    except Exception:
        return None

def build_github_analysis(username: str, api_key: str) -> dict:
    try:
        profile_response = fetch_github_profile(username)
        if not profile_response.get("ok"):
            return {"ok": False, "message": "GitHub analysis unavailable."}

        repo_response = fetch_github_repositories(username)
        repositories = repo_response.get("repositories", []) if repo_response.get("ok") else []
        profile = profile_response.get("profile", {})
        analysis = generate_github_ai_analysis(profile, repositories, api_key) if api_key else None

        return {
            "ok": True,
            "profile": profile,
            "repositories": repositories,
            "languages": summarize_github_languages(repositories),
            "analysis": analysis,
        }
    except Exception:
        return {"ok": False, "message": "GitHub analysis unavailable."}

def render_github_insights(github_data: Optional[dict]) -> str:
    if not github_data:
        return '<div class="cc-divider"></div><div class="cc-section-title">GitHub Intelligence</div><div class="cc-summary">No GitHub profile available.</div>'
    if not github_data.get("ok"):
        return '<div class="cc-divider"></div><div class="cc-section-title">GitHub Intelligence</div><div class="cc-summary">GitHub analysis unavailable.</div>'

    profile = github_data.get("profile", {})
    analysis = github_data.get("analysis") or {}
    languages = github_data.get("languages", {})
    strengths = analysis.get("technical_strengths") or []
    language_items = "".join(f"<li>{html.escape(str(lang))}</li>" for lang in list(languages.keys())[:3]) or "<li>Unknown</li>"
    strength_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in strengths[:3]) or "<li>Not available</li>"
    github_score = html.escape(str(analysis.get("github_score", "N/A")))
    activity = html.escape(str(analysis.get("activity_level", "N/A")))
    maturity = html.escape(str(analysis.get("open_source_maturity", "N/A")))
    summary = html.escape(str(analysis.get("summary", "GitHub profile collected. AI analysis unavailable.")))
    username = html.escape(str(profile.get("username", "")))

    return f"""
    <div class="cc-divider"></div>
    <div class="cc-section-title">GitHub Intelligence</div>
    <div class="cc-details">
        <div>
            <div class="cc-subtitle">GitHub Score</div>
            <div class="metric-value" style="font-size:24px;">{github_score} / 100</div>
            <div class="cc-subtitle">Activity</div>
            <div class="cc-summary">{activity}</div>
            <div class="cc-section-title" style="margin-top:16px;">Top Languages</div>
            <ul class="cc-list">{language_items}</ul>
        </div>
        <div>
            <div class="cc-section-title">Technical Strengths</div>
            <ul class="cc-list">{strength_items}</ul>
            <div class="cc-section-title" style="margin-top:16px;">Open Source</div>
            <div class="cc-summary">{maturity}</div>
        </div>
    </div>
    <div class="cc-section-title" style="margin-top:16px;">Summary</div>
    <div class="cc-summary">{summary}</div>
    <div class="cc-subtitle" style="margin-top:12px;">GitHub: {username}</div>
    """

def render_github_details(github_data: Optional[dict]):
    if not github_data:
        st.info("No GitHub profile available.")
        return
    if not github_data.get("ok"):
        st.warning("GitHub analysis unavailable.")
        return

    profile = github_data.get("profile", {})
    repositories = github_data.get("repositories", [])
    languages = github_data.get("languages", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Followers", profile.get("followers", 0))
    c2.metric("Following", profile.get("following", 0))
    c3.metric("Public Repos", profile.get("public_repositories", 0))
    st.caption(f"Last activity: {profile.get('updated_at', 'N/A')}")

    if languages:
        st.write("Language breakdown")
        st.dataframe(pd.DataFrame([{"Language": k, "Repositories": v} for k, v in languages.items()]), use_container_width=True, hide_index=True)

    if repositories:
        st.write("Top repositories")
        st.dataframe(pd.DataFrame(repositories), use_container_width=True, hide_index=True)
    else:
        st.info("No public repositories found.")


# ---------------------------------------------------------------------------
# Candidate Assessment Module — External Assessment Scores
# ---------------------------------------------------------------------------

ASSESSMENT_COLUMN_PATTERNS = {
    "coding": ["coding score", "coding test", "coding", "coding_score", "coding_test", "coding assessment"],
    "aptitude": ["aptitude score", "aptitude test", "aptitude", "aptitude_score", "aptitude_test", "aptitude assessment"],
    "logical": ["logical score", "logical reasoning", "logical", "logical_score", "logical_reasoning", "logical test"],
    "communication": ["communication score", "communication assessment", "communication", "communication_score", "communication_assessment", "communication test"],
    "email": ["email", "e-mail", "email address", "email_address", "email_id", "emailid", "mail"],
    "candidate_name": ["candidate name", "candidate_name", "name", "full_name", "full name", "applicant name", "applicant", "candidate"],
}

def _find_assessment_column(df: pd.DataFrame, patterns: list) -> Optional[str]:
    lower_map = {col.lower().strip(): col for col in df.columns}
    for c in patterns:
        if c in lower_map:
            return lower_map[c]
    return None


def process_assessment_csv(assess_df: pd.DataFrame, candidates_df: Optional[pd.DataFrame]) -> tuple[dict, dict]:
    try:
        test_results = {}
        detected_cols = []

        col_map = {}
        for key in ["coding", "aptitude", "logical", "communication"]:
            col = _find_assessment_column(assess_df, ASSESSMENT_COLUMN_PATTERNS[key])
            if col:
                col_map[key] = col
                detected_cols.append(key.capitalize())

        email_col = _find_assessment_column(assess_df, ASSESSMENT_COLUMN_PATTERNS["email"])
        name_col = _find_assessment_column(assess_df, ASSESSMENT_COLUMN_PATTERNS["candidate_name"])

        matched = 0
        unmatched = 0

        if candidates_df is not None:
            cand_name_col = find_name_column(candidates_df)
            cand_email_col = _find_assessment_column(candidates_df, ASSESSMENT_COLUMN_PATTERNS["email"])

            email_lookup = {}
            name_lookup = {}
            if email_col and cand_email_col:
                for cidx, crow in candidates_df.iterrows():
                    cand_email = str(crow[cand_email_col]).strip().lower() if pd.notna(crow[cand_email_col]) else ""
                    if cand_email:
                        email_lookup[cand_email] = cidx
            if name_col and cand_name_col:
                for cidx, crow in candidates_df.iterrows():
                    cand_name = str(crow[cand_name_col]).strip().lower() if pd.notna(crow[cand_name_col]) else ""
                    if cand_name:
                        name_lookup[cand_name] = cidx

            for _, arow in assess_df.iterrows():
                match_idx = None

                if email_col and cand_email_col:
                    assess_email = str(arow[email_col]).strip().lower() if pd.notna(arow[email_col]) else ""
                    if assess_email:
                        match_idx = email_lookup.get(assess_email)

                if match_idx is None and name_col and cand_name_col:
                    assess_name = str(arow[name_col]).strip().lower() if pd.notna(arow[name_col]) else ""
                    if assess_name:
                        match_idx = name_lookup.get(assess_name)

                if match_idx is not None:
                    scores = {}
                    for key in ["coding", "aptitude", "logical", "communication"]:
                        if key in col_map:
                            val = arow[col_map[key]]
                            try:
                                scores[key] = int(float(str(val).strip())) if pd.notna(val) else None
                            except (ValueError, TypeError):
                                scores[key] = None
                    test_results[match_idx] = scores
                    matched += 1
                else:
                    unmatched += 1
        else:
            unmatched = len(assess_df)

        stats = {
            "total": len(assess_df),
            "matched": matched,
            "unmatched": unmatched,
            "detected": detected_cols,
        }

        return test_results, stats
    except Exception:
        return {}, {"total": len(assess_df) if assess_df is not None else 0, "matched": 0, "unmatched": 0, "detected": []}

def calculate_combined_score(ai_score: int, test_data: Optional[dict], weights: dict) -> Optional[int]:
    if not test_data:
        return None

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return None

    score = ai_score * (weights.get("ai", 0) / total_weight)
    for key in ["coding", "aptitude", "logical", "communication"]:
        val = test_data.get(key)
        if val is not None:
            score += int(val) * (weights.get(key, 0) / total_weight)

    return max(0, min(100, int(round(score))))

def render_assessment_insights(test_data: Optional[dict], combined_score: Optional[int]) -> str:
    if not test_data:
        return '<div class="cc-divider"></div><div class="cc-section-title">Assessment Results</div><div class="cc-summary">No assessment data uploaded.</div>'

    items = ""
    label_map = {"coding": "Coding Test", "aptitude": "Aptitude", "logical": "Logical", "communication": "Communication"}
    for key, label in label_map.items():
        val = test_data.get(key)
        if val is not None:
            items += f'<div style="margin-top:8px;"><div class="cc-subtitle">{label}</div><div style="font-size:20px;font-weight:600;">{int(val)}</div></div>'

    combined_html = ""
    if combined_score is not None:
        combined_html = f'<div style="margin-top:16px;"><div class="cc-section-title" style="margin-bottom:4px;">Combined Recruiter Score</div><div style="font-size:24px;font-weight:600;">{combined_score} / 100</div></div>'

    return f"""
    <div class="cc-divider"></div>
    <div class="cc-section-title">Assessment Results</div>
    {items}
    {combined_html}
    """


# ---------------------------------------------------------------------------
# Interview Scheduling Module
# ---------------------------------------------------------------------------

def generate_meeting_link(mode: str) -> Optional[str]:
    if mode == "google_meet":
        return None
    return None
def generate_calendar_event(schedule: dict) -> Optional[dict]:
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        import datetime

        if "google_calendar_creds" in st.secrets:
            creds = Credentials.from_authorized_user_info(
                st.secrets["google_calendar_creds"]
            )

            if creds and creds.valid:
                service = build("calendar", "v3", credentials=creds)

                # Build start time
                start_dt = datetime.datetime.fromisoformat(
                    f"{schedule.get('date')}T{schedule.get('time')}"
                )

                # Calculate end time using interview duration
                duration = int(schedule.get("duration", 60))
                end_dt = start_dt + datetime.timedelta(minutes=duration)

                event = {
                    "summary": f"Interview - {schedule.get('type', '')}",
                    "description": schedule.get("notes", ""),
                    "start": {
                        "dateTime": start_dt.isoformat(),
                        "timeZone": "UTC",
                    },
                    "end": {
                        "dateTime": end_dt.isoformat(),
                        "timeZone": "UTC",
                    },
                }

                result = service.events().insert(
                    calendarId="primary",
                    body=event
                ).execute()

                return {
                    "id": result.get("id"),
                    "link": result.get("htmlLink"),
                }

    except Exception:
        pass

    return None

def create_interview_schedule(candidate_idx: int, data: dict) -> dict:
    return {
        "candidate_idx": candidate_idx,
        "date": data.get("date", ""),
        "time": data.get("time", ""),
        "duration": data.get("duration", 60),
        "type": data.get("type", "Technical"),
        "interviewer": data.get("interviewer", ""),
        "notes": data.get("notes", ""),
        "location": data.get("location", ""),
        "mode": data.get("mode", "Online"),
        "meeting_link": data.get("meeting_link", ""),
        "status": data.get("status", "Scheduled"),
        "calendar_event": data.get("calendar_event"),
    }

def render_schedule_status(schedule: Optional[dict]) -> str:
    if not schedule:
        return ""
    status = schedule.get("status", "")
    if status == "Scheduled":
        sb = "badge badge-success"
    elif status == "Completed":
        sb = "badge badge-neutral"
    elif status == "Cancelled":
        sb = "badge badge-danger"
    else:
        sb = "badge badge-warning"
    date_val = html.escape(str(schedule.get("date", "")))
    time_val = html.escape(str(schedule.get("time", "")))
    mode_val = html.escape(str(schedule.get("mode", "")))
    interviewer_val = html.escape(str(schedule.get("interviewer", "")))
    link = html.escape(str(schedule.get("meeting_link", "")))
    link_html = f'<div class="cc-subtitle" style="margin-top:8px;">Meeting Link</div><div class="cc-summary"><a href="{link}" target="_blank">{link}</a></div>' if link else ""
    return f"""
    <div class="cc-divider"></div>
    <div class="cc-section-title">Interview Status</div>
    <div><span class="{sb}">{html.escape(status)}</span></div>
    <div style="margin-top:8px;">
        <div class="cc-subtitle">Date & Time</div>
        <div class="cc-summary">{date_val} at {time_val}</div>
        <div class="cc-subtitle">Mode</div>
        <div class="cc-summary">{mode_val}</div>
        <div class="cc-subtitle">Interviewer</div>
        <div class="cc-summary">{interviewer_val}</div>
        {link_html}
    </div>
    """

def render_interview_panel():
    st.markdown('<div class="cc-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Interview Scheduling</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle" style="margin-bottom:16px;">Schedule interviews for evaluated candidates.</div>', unsafe_allow_html=True)

    df = st.session_state.candidates_df
    results = st.session_state.evaluation_results
    if df is None or not results:
        st.markdown('<div class="empty-state">No candidates available. Run evaluation first.</div>', unsafe_allow_html=True)
        return

    name_col = find_name_column(df)
    candidate_labels = []
    candidate_map = {}
    for idx in sorted(results.keys()):
        idx = int(idx)
        if idx < len(df):
            row = df.iloc[idx]
            label = str(row.get(name_col, f"Candidate {idx + 1}"))
            candidate_labels.append(label)
            candidate_map[label] = idx

    if not candidate_labels:
        return

    selected_label = st.selectbox("Select Candidate", options=candidate_labels, key="schedule_candidate_select")
    selected_idx = candidate_map[selected_label]
    existing = st.session_state.interview_schedule.get(selected_idx, {})
    today = datetime.date.today()

    col1, col2 = st.columns(2)
    with col1:
        d = st.date_input("Interview Date", value=today, key="schedule_date")
        t = st.time_input("Interview Time", value=datetime.time(10, 0), key="schedule_time")
        dur = st.selectbox("Duration (minutes)", [15, 30, 45, 60, 90, 120], index=3, key="schedule_duration")
        itype = st.selectbox("Interview Type", ["Technical", "HR", "Final", "Managerial"], key="schedule_type")
    with col2:
        inv = st.text_input("Interviewer Name", value=existing.get("interviewer", ""), key="schedule_interviewer")
        notes = st.text_area("Meeting Notes", value=existing.get("notes", ""), height=100, key="schedule_notes")
        loc = st.text_input("Meeting Location", value=existing.get("location", ""), key="schedule_location")
        current_mode_idx = 0
        if existing.get("mode") in ["Online", "Offline", "Hybrid"]:
            current_mode_idx = ["Online", "Offline", "Hybrid"].index(existing.get("mode", "Online"))
        mode = st.selectbox("Meeting Mode", ["Online", "Offline", "Hybrid"], index=current_mode_idx, key="schedule_mode")

    meeting_link = existing.get("meeting_link", "")
    if mode == "Online":
        c1, c2 = st.columns(2)
        with c1:
            gen_meet = st.checkbox("Generate Google Meet Link (Requires Google Credentials)", value=False, key="gen_meet_link")
        with c2:
            custom_url = st.text_input("Or enter Custom Meeting URL", value=meeting_link, key="custom_meet_url")
        if gen_meet:
            generated = generate_meeting_link("google_meet")
            if generated:
                meeting_link = generated
        elif custom_url:
            meeting_link = custom_url

    existing_schedule = st.session_state.interview_schedule.get(selected_idx)
    if existing_schedule and existing_schedule.get("status") in ("Scheduled", "Completed", "Cancelled"):
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="cc-section-title">Current Status: <span class="{"badge-success" if existing_schedule.get("status") == "Scheduled" else "badge-neutral" if existing_schedule.get("status") == "Completed" else "badge-danger"}">{existing_schedule.get("status")}</span></div>', unsafe_allow_html=True)
        action_cols = st.columns(2)

        with action_cols[0]:
            if st.button("Cancel Interview", key=f"cancel_{selected_idx}", type="secondary"):
                st.session_state.interview_schedule[selected_idx]["status"] = "Cancelled"
                st.rerun()

        with action_cols[1]:
            if st.button("Mark Completed", key=f"complete_{selected_idx}", type="secondary"):
                st.session_state.interview_schedule[selected_idx]["status"] = "Completed"
                st.rerun()
    if st.button("Schedule Interview", type="primary", key="schedule_submit"):
        schedule = create_interview_schedule(selected_idx, {
            "date": str(d), "time": str(t), "duration": dur,
            "type": itype, "interviewer": inv, "notes": notes,
            "location": loc, "mode": mode, "meeting_link": meeting_link,
            "status": "Scheduled"
        })
        cal = generate_calendar_event(schedule)
        if cal:
            schedule["calendar_event"] = cal
        st.session_state.interview_schedule[selected_idx] = schedule
        st.success(f"Interview scheduled for {selected_label}")
        st.rerun()


# ===========================================================================
# AI Recruiter Copilot Module
# ===========================================================================
"""Comprehensive hiring dashboard for recruiters. Consumes evaluation data
without modifying it."""

from collections import Counter

def safe_get(d: dict, key: str, default: str = "") -> str:
    if not isinstance(d, dict): return default
    v = d.get(key, default)
    return v if v is not None else default

def safe_list_get(d: dict, key: str, default: list = None) -> list:
    if default is None: default = []
    if not isinstance(d, dict): return default
    v = d.get(key, default)
    return v if isinstance(v, list) else default

def compute_pool_summary() -> dict:
    df = st.session_state.candidates_df
    results = st.session_state.evaluation_results
    interview = st.session_state.interview_schedule
    scores = [r.get("score", 0) for r in results.values()]
    recs = [r.get("recommendation", "") for r in results.values()]
    return {
        "total": len(df) if df is not None else 0,
        "evaluated": len(results),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
        "shortlisted": sum(1 for r in recs if normalize_recommendation(r) == "Shortlist"),
        "maybe": sum(1 for r in recs if normalize_recommendation(r) == "Maybe"),
        "rejected": sum(1 for r in recs if normalize_recommendation(r) == "Reject"),
        "scores": scores,
        "highest_score": max(scores) if scores else 0,
        "lowest_score": min(scores) if scores else 0,
        "score_std": round(pd.Series(scores).std(), 1) if len(scores) > 1 else 0,
        "scheduled": sum(1 for s in interview.values() if s.get("status") == "Scheduled"),
    }

def compute_skill_gap_analysis() -> dict:
    results = st.session_state.evaluation_results
    strengths_counter: Counter = Counter()
    weaknesses_counter: Counter = Counter()
    for r in results.values():
        for s in safe_list_get(r, "strengths", safe_list_get(r, "Strengths", [])):
            strengths_counter[s.strip()] += 1
        for w in safe_list_get(r, "weaknesses", safe_list_get(r, "Weaknesses", [])):
            weaknesses_counter[w.strip()] += 1
    github = st.session_state.github_analysis or {}
    for g in github.values():
        ga = g.get("analysis") or {}
        for s in safe_list_get(ga, "technical_strengths", []):
            strengths_counter[s.strip()] += 1
    return {
        "common_skills": [{"skill": s, "count": c} for s, c in strengths_counter.most_common(15)],
        "missing_skills": [{"skill": s, "count": c} for s, c in weaknesses_counter.most_common(15)],
        "skill_gaps": [{"skill": s, "missing_count": c, "coverage": strengths_counter.get(s, 0)}
                       for s, c in weaknesses_counter.most_common(20)],
    }

def get_top_candidates(n: int = 5) -> list:
    df = st.session_state.candidates_df
    results = st.session_state.evaluation_results
    test = st.session_state.test_results or {}
    weights = st.session_state.assessment_weights
    name_col = find_name_column(df) if df is not None else None
    name_col = name_col if name_col else "Name"
    items = []
    for idx, r in results.items():
        idx = int(idx)
        if df is not None and idx >= len(df): continue
        row = df.iloc[idx] if df is not None else None
        name = str(row.get(name_col, f"Candidate {idx+1}")) if row is not None else f"Candidate {idx+1}"
        score = r.get("score", 0)
        combined = calculate_combined_score(score, test.get(idx, {}), weights)
        items.append({"idx": idx, "name": name, "score": score,
                       "combined_score": combined,
                       "recommendation": normalize_recommendation(r.get("recommendation", "")),
                       "strengths": safe_list_get(r, "strengths", safe_list_get(r, "Strengths", [])),
                       "weaknesses": safe_list_get(r, "weaknesses", safe_list_get(r, "Weaknesses", [])),
                       "summary": r.get("summary", "")})
    items.sort(key=lambda c: (c["combined_score"] if c["combined_score"] is not None else c["score"]),
               reverse=True)
    return items[:n]

def _build_candidates_json() -> str:
    df = st.session_state.candidates_df
    results = st.session_state.evaluation_results
    test = st.session_state.test_results or {}
    weights = st.session_state.assessment_weights
    name_col = find_name_column(df) if df is not None else None
    name_col = name_col if name_col else "Name"
    if df is None or not results: return "[]"
    items = []
    for idx, r in results.items():
        idx = int(idx)
        if idx >= len(df): continue
        row = df.iloc[idx]
        name = str(row.get(name_col, f"Candidate {idx+1}"))
        t_data = test.get(idx, {})
        combined = calculate_combined_score(r.get("score", 0), t_data, weights)
        items.append({"name": name, "ai_score": r.get("score", 0),
                       "recommendation": normalize_recommendation(r.get("recommendation", "")),
                       "strengths": safe_list_get(r, "strengths", safe_list_get(r, "Strengths", [])),
                       "weaknesses": safe_list_get(r, "weaknesses", safe_list_get(r, "Weaknesses", [])),
                       "combined_score": combined})
    return json.dumps(items, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Gemini cached helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_hiring_summary(_candidates_json: str, _jd: str, _cache_key: int) -> dict:
    if not st.session_state.current_api_key: return {"error": "No API key"}
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=st.session_state.current_api_key)
        prompt = f"""You are an expert technical recruiter. Analyze this candidate pool for the job below.
Return JSON: overall_quality (Excellent/Good/Average/Below Average/Poor), hiring_confidence (0-100),
summary (2-3 sentences), strength, weakness, recommended_action.
Job: {(_jd or "")[:2000]}
Candidates: {_candidates_json[:15000]}"""
        resp = client.models.generate_content(
            model="gemini-2.0-flash-lite", contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"))
        r = json.loads(resp.text)
        return {k: r.get(k, "N/A") for k in
                ["overall_quality", "hiring_confidence", "summary", "strength", "weakness", "recommended_action"]}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(show_spinner=False)
def _cached_risks(_candidates_json: str, _cache_key: int) -> list:
    if not st.session_state.current_api_key: return [{"risk": "No API key", "severity": "medium", "mitigation": ""}]
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=st.session_state.current_api_key)
        prompt = f"""Analyze candidate pool for hiring risks. Return JSON array of {{risk, severity (high/medium/low), mitigation}}.
Candidates: {_candidates_json[:10000]}"""
        resp = client.models.generate_content(
            model="gemini-2.0-flash-lite", contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"))
        r = json.loads(resp.text)
        return r[:7] if isinstance(r, list) else []
    except Exception as e:
        return [{"risk": str(e), "severity": "medium", "mitigation": "Manual review"}]

@st.cache_data(show_spinner=False)
def _cached_actions(_candidates_json: str, _jd: str, _cache_key: int) -> list:
    if not st.session_state.current_api_key: return [{"action": "No API key", "priority": "high", "detail": ""}]
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=st.session_state.current_api_key)
        prompt = f"""Generate 3-5 actionable next steps for the recruiter based on this data.
Return JSON array of {{action, priority (high/medium/low), detail}}.
Job: {(_jd or "")[:1000]}
Candidates: {_candidates_json[:10000]}"""
        resp = client.models.generate_content(
            model="gemini-2.0-flash-lite", contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"))
        r = json.loads(resp.text)
        return r[:5] if isinstance(r, list) else []
    except Exception as e:
        return [{"action": str(e), "priority": "medium", "detail": ""}]

@st.cache_data(show_spinner=False)
def _cached_cutoff(_candidates_json: str, _cache_key: int) -> dict:
    if not st.session_state.current_api_key: return {"error": "No API key"}
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=st.session_state.current_api_key)
        prompt = f"""Analyze candidate pool and suggest interview cutoff score.
Return JSON: suggested_cutoff (0-100), candidates_above (int), candidates_below (int), rationale, recommended_interview_slots (int).
Candidates: {_candidates_json[:10000]}"""
        resp = client.models.generate_content(
            model="gemini-2.0-flash-lite", contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"))
        r = json.loads(resp.text)
        return {"suggested_cutoff": r.get("suggested_cutoff", 70),
                "candidates_above": r.get("candidates_above", 0),
                "candidates_below": r.get("candidates_below", 0),
                "rationale": r.get("rationale", ""),
                "recommended_interview_slots": r.get("recommended_interview_slots", 0)}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(show_spinner=False)
def _cached_compare(_a_json: str, _b_json: str, _cache_key: int) -> dict:
    if not st.session_state.current_api_key: return {"error": "No API key"}
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=st.session_state.current_api_key)
        prompt = f"""Compare these two candidates side-by-side.
Return JSON: recommendation (1-2 sentences), strengths_a (list), strengths_b (list),
weaknesses_a (list), weaknesses_b (list), fit_score_a (0-100), fit_score_b (0-100).
Candidate A: {_a_json[:5000]}
Candidate B: {_b_json[:5000]}"""
        resp = client.models.generate_content(
            model="gemini-2.0-flash-lite", contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"))
        r = json.loads(resp.text)
        return {"recommendation": safe_get(r, "recommendation", ""),
                "strengths_a": safe_list_get(r, "strengths_a", []),
                "strengths_b": safe_list_get(r, "strengths_b", []),
                "weaknesses_a": safe_list_get(r, "weaknesses_a", []),
                "weaknesses_b": safe_list_get(r, "weaknesses_b", []),
                "fit_score_a": r.get("fit_score_a", "N/A"),
                "fit_score_b": r.get("fit_score_b", "N/A")}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(show_spinner=False)
def _cached_top_reasons(_candidates_json: str, _cache_key: int) -> list:
    if not st.session_state.current_api_key: return []
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=st.session_state.current_api_key)
        prompt = f"""For each candidate provide one sentence explaining why they are a top candidate.
Return JSON array of {{name, reason}}.
Candidates: {_candidates_json[:8000]}"""
        resp = client.models.generate_content(
            model="gemini-2.0-flash-lite", contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"))
        r = json.loads(resp.text)
        return r if isinstance(r, list) else []
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def render_candidate_pool_summary(s: dict):
    st.markdown('<div class="section-subtitle">Pool Overview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total / Evaluated", f'{s.get("total",0)} / {s.get("evaluated",0)}')
    with c2: st.metric("Average Score", f'{s.get("avg_score",0)}%')
    with c3: st.metric("Score Range", f'{s.get("lowest_score",0)}% - {s.get("highest_score",0)}%')
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Shortlist", s.get("shortlisted",0))
    with c2: st.metric("Maybe", s.get("maybe",0))
    with c3: st.metric("Reject", s.get("rejected",0))
    with c4: st.metric("Scheduled", s.get("scheduled",0))

def render_skill_gap_analysis(skills: dict):
    st.markdown('<div class="section-subtitle">Skills Overview</div>', unsafe_allow_html=True)
    if not skills.get("common_skills") and not skills.get("missing_skills"):
        st.info("No skill data yet. Evaluate candidates to populate.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Strengths**")
        if skills.get("common_skills"):
            df = pd.DataFrame(skills["common_skills"])
            if not df.empty: st.bar_chart(df.set_index("skill")["count"])
    with c2:
        st.markdown("**Most Missing Skills**")
        if skills.get("missing_skills"):
            df = pd.DataFrame(skills["missing_skills"])
            if not df.empty: st.bar_chart(df.set_index("skill")["count"])
    if skills.get("skill_gaps"):
        st.markdown("**Skill Gap Detail**")
        gap_df = pd.DataFrame(skills["skill_gaps"])
        gap_df.columns = ["Skill", "Candidates Missing It", "Candidates Having It"]
        st.dataframe(gap_df, use_container_width=True, hide_index=True)

def render_candidate_distribution():
    st.markdown('<div class="section-subtitle">Score Distribution</div>', unsafe_allow_html=True)
    results = st.session_state.evaluation_results
    if not results:
        st.info("No evaluations yet.")
        return
    scores = pd.DataFrame({"Candidate": list(results.keys()), "Score": [r.get("score",0) for r in results.values()]})
    c1, c2 = st.columns(2)
    with c1: st.bar_chart(scores.set_index("Candidate")["Score"])
    with c2:
        rec = pd.Series([normalize_recommendation(r.get("recommendation","")) for r in results.values()]).value_counts()
        st.bar_chart(rec)

def render_top_candidates(top: list):
    st.markdown('<div class="section-subtitle">Top Candidates</div>', unsafe_allow_html=True)
    if not top:
        st.info("No candidates evaluated.")
        return
    for i, c in enumerate(top[:5], 1):
        combined = c.get("combined_score")
        score_str = f'{c.get("score", 0)}%'
        combined_str = f' | Combined: {combined:.0f}%' if combined is not None else ''
        reason = c.get("reason", "")
        r_html = f'<div style="color:#8b8fa3;font-size:0.9em;margin-top:4px;">{reason}</div>' if reason else ''
        strength_items = "".join(f"<li>{s}</li>" for s in c.get("strengths", []))
        weakness_items = "".join(f"<li>{w}</li>" for w in c.get("weaknesses", []))
        st.markdown(f"""<div class="candidate-card"><div class="cc-header"><div><div class="cc-title">#{i} {c.get("name", "Unknown")}</div><div class="cc-subtitle">{score_str}{combined_str} | {normalize_recommendation(c.get("recommendation",""))}</div></div></div>{r_html}<div class="cc-divider"></div><div class="cc-details"><div><div class="cc-section-title">Strengths</div><ul class="cc-list">{strength_items}</ul></div><div><div class="cc-section-title">Areas to Probe</div><ul class="cc-list">{weakness_items}</ul></div></div></div>""", unsafe_allow_html=True)

def render_ai_hiring_recommendation(summary: dict):
    st.markdown('<div class="section-subtitle">AI Recommendation</div>', unsafe_allow_html=True)
    if "error" in summary:
        st.error(summary["error"]); return
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Overall Quality", safe_get(summary,"overall_quality","N/A"))
    with c2: st.metric("Hiring Confidence", safe_get(summary,"hiring_confidence","N/A"))
    with c3: st.metric("Recommended Action", safe_get(summary,"recommended_action","N/A"))
    st.markdown(f"**Summary:** {safe_get(summary,'summary','')}")
    st.markdown(f"**Key Strength:** {safe_get(summary,'strength','')}  |  **Key Weakness:** {safe_get(summary,'weakness','')}")

def render_candidate_comparison(cache_key: int):
    st.markdown('<div class="section-subtitle">Side-by-Side</div>', unsafe_allow_html=True)
    df = st.session_state.candidates_df
    results = st.session_state.evaluation_results
    name_col = find_name_column(df) if df is not None else None
    name_col = name_col if name_col else "Name"
    if not results or df is None:
        st.info("Evaluate candidates first."); return
    options = {}
    for idx, r in results.items():
        idx = int(idx)
        if idx < len(df):
            name = str(df.iloc[idx].get(name_col, f"Candidate {idx+1}"))
            options[f"{name} (Score: {r.get('score',0)}%)"] = (idx, name)
    if len(options) < 2:
        st.info("Need at least 2 candidates."); return
    opt_list = list(options.keys())
    col1, col2 = st.columns(2)
    with col1: a_sel = st.selectbox("Candidate A", opt_list, key="comp_a")
    with col2: b_sel = st.selectbox("Candidate B", opt_list, key="comp_b", index=min(1,len(opt_list)-1))
    if st.button("Compare", type="primary", key="compare_btn"):
        if a_sel == b_sel: st.warning("Select different candidates."); return
        a_idx, a_n = options[a_sel]; b_idx, b_n = options[b_sel]
        a = results[a_idx]; b = results[b_idx]
        a_json = json.dumps({"name": a_n, "score": a.get("score",0), "strengths": safe_list_get(a,"strengths",safe_list_get(a,"Strengths",[])), "weaknesses": safe_list_get(a,"weaknesses",safe_list_get(a,"Weaknesses",[]))})
        b_json = json.dumps({"name": b_n, "score": b.get("score",0), "strengths": safe_list_get(b,"strengths",safe_list_get(b,"Strengths",[])), "weaknesses": safe_list_get(b,"weaknesses",safe_list_get(b,"Weaknesses",[]))})
        with st.spinner("Comparing..."):
            comp = _cached_compare(a_json, b_json, cache_key)
        if "error" in comp: st.error(comp["error"]); return
        st.markdown(f"**Recommendation:** {comp.get('recommendation','')}")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"**{a_n} — Fit: {comp.get('fit_score_a','N/A')}**")
            for s in safe_list_get(comp,"strengths_a",[]): st.markdown(f"- ✅ {s}")
            for w in safe_list_get(comp,"weaknesses_a",[]): st.markdown(f"- ⚠️ {w}")
        with cc2:
            st.markdown(f"**{b_n} — Fit: {comp.get('fit_score_b','N/A')}**")
            for s in safe_list_get(comp,"strengths_b",[]): st.markdown(f"- ✅ {s}")
            for w in safe_list_get(comp,"weaknesses_b",[]): st.markdown(f"- ⚠️ {w}")

def render_interview_cutoff(cutoff: dict):
    st.markdown('<div class="section-subtitle">Suggested Cutoff</div>', unsafe_allow_html=True)
    if "error" in cutoff: st.error(cutoff["error"]); return
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Cutoff Score", f'{cutoff.get("suggested_cutoff","N/A")}%')
    with c2: st.metric("Pass", cutoff.get("candidates_above","N/A"))
    with c3: st.metric("Below", cutoff.get("candidates_below","N/A"))
    st.markdown(f"**Rationale:** {cutoff.get('rationale','')}")
    if cutoff.get("recommended_interview_slots"):
        st.markdown(f"**Recommended Slots:** {cutoff['recommended_interview_slots']}")

def render_hiring_risks(risks: list):
    st.markdown('<div class="section-subtitle">Hiring Risks</div>', unsafe_allow_html=True)
    if not risks: return
    for r in risks:
        sev = r.get("severity","medium").lower()
        icon = {"high":"🔴","medium":"🟡","low":"🟢"}.get(sev,"🟡")
        st.markdown(f"{icon} **[{sev.upper()}]** {r.get('risk','')}")
        if r.get("mitigation"): st.markdown(f"   *Mitigation:* {r['mitigation']}")
        st.divider()

def render_recruiter_actions(actions: list):
    st.markdown('<div class="section-subtitle">Recommended Actions</div>', unsafe_allow_html=True)
    if not actions: return
    for i, a in enumerate(actions, 1):
        pri = a.get("priority","medium").lower()
        icon = {"high":"🔴","medium":"🟡","low":"🟢"}.get(pri,"🟡")
        st.markdown(f"{i}. {icon} **[{pri.upper()}]** {a.get('action','')}")
        if a.get("detail"): st.markdown(f"   *{a['detail']}*")
        st.divider()

def render_export_report(pdf_bytes: Optional[bytes], filename: str):
    st.markdown('<div class="section-subtitle">Export</div>', unsafe_allow_html=True)
    if pdf_bytes:
        st.download_button("Download Recruiter Report (PDF)", pdf_bytes, file_name=filename, mime="application/pdf", type="primary", use_container_width=True)
    else:
        st.warning("Install reportlab (`pip install reportlab`) to enable PDF export.")

def generate_recruiter_pdf(summary: dict, skills: dict, top: list, risks: list, actions: list, cutoff: dict, hiring: dict) -> Optional[bytes]:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph("AI Recruiter Copilot Report", styles["Title"]), Spacer(1, 12)]
        if hiring and "error" not in hiring:
            story.append(Paragraph(f"Overall Quality: {hiring.get('overall_quality','N/A')} | Confidence: {hiring.get('hiring_confidence','N/A')}", styles["Normal"]))
            story.append(Paragraph(f"Summary: {hiring.get('summary','')}", styles["Normal"]))
        story += [Spacer(1,16), Paragraph("Pool Summary", styles["Heading2"]),
                  Paragraph(f"Total: {summary.get('total',0)} | Evaluated: {summary.get('evaluated',0)} | Avg: {summary.get('avg_score',0)}%", styles["Normal"]),
                  Paragraph(f"Shortlist: {summary.get('shortlisted',0)} | Maybe: {summary.get('maybe',0)} | Reject: {summary.get('rejected',0)}", styles["Normal"])]
        story += [Spacer(1,16), Paragraph("Top Candidates", styles["Heading2"])]
        for i, c in enumerate(top[:5], 1):
            cs = c.get("combined_score", c.get("score","N/A"))
            story.append(Paragraph(f"{i}. {c['name']} — Score: {c['score']}, Combined: {cs}", styles["Normal"]))
        if cutoff and "error" not in cutoff:
            story += [Spacer(1,16), Paragraph("Interview Cutoff", styles["Heading2"]),
                      Paragraph(f"Suggested: {cutoff.get('suggested_cutoff','N/A')}% | Pass: {cutoff.get('candidates_above','N/A')} | Below: {cutoff.get('candidates_below','N/A')}", styles["Normal"]),
                      Paragraph(f"Rationale: {cutoff.get('rationale','')}", styles["Normal"])]
        if risks:
            story += [Spacer(1,16), Paragraph("Hiring Risks", styles["Heading2"])]
            for r in risks[:5]:
                story.append(Paragraph(f"- [{r.get('severity','').upper()}] {r.get('risk','')}", styles["Normal"]))
        if actions:
            story += [Spacer(1,16), Paragraph("Recommended Actions", styles["Heading2"])]
            for a in actions[:5]:
                story.append(Paragraph(f"- [{a.get('priority','').upper()}] {a.get('action','')}", styles["Normal"]))
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    except ImportError:
        return None
    except Exception:
        return None

def _show_rule_based_warnings(summary: dict):
    if summary.get("evaluated",0) == 0: return
    warns = []
    if summary.get("avg_score",100) < 50:
        warns.append("Low average score. Consider revising JD or sourcing channels.")
    if summary.get("rejected",0) > summary.get("evaluated",1) * 0.5:
        warns.append("Over 50% are 'Reject'. Review screening criteria.")
    if summary.get("score_std",10) < 5 and summary.get("evaluated",0) > 3:
        warns.append("Very low score variance. Candidates may not be differentiated.")
    if summary.get("shortlisted",0) > 0 and summary.get("scheduled",0) == 0:
        warns.append(f"{summary.get('shortlisted', 0)} Shortlist candidate(s) but none scheduled for interview.")
    if warns:
        with st.expander("Rule-Based Warnings", expanded=False):
            for w in warns: st.warning(w)

# ---------------------------------------------------------------------------
# Main Copilot Page
# ---------------------------------------------------------------------------

def render_recruiter_copilot():
    st.markdown('<div class="page-title">AI Recruiter Copilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">AI-powered insights, comparisons, and recommendations for your hiring pipeline.</div>', unsafe_allow_html=True)
    df, results = st.session_state.candidates_df, st.session_state.evaluation_results
    if df is None or not results:
        st.info("Upload and evaluate candidates to activate the Recruiter Copilot.")
        return
    cache_key = st.session_state.get("copilot_cache_key", 0)
    pool = compute_pool_summary()
    skills = compute_skill_gap_analysis()
    top = get_top_candidates(5)
    data_json = _build_candidates_json()
    jd = st.session_state.job_description

    refresh_cols = st.columns([6, 1])
    with refresh_cols[1]:
        if st.button("Refresh Insights", type="primary", use_container_width=True):
            st.session_state.copilot_cache_key = cache_key + 1
            st.rerun()

    with st.expander("Candidate Pool Summary", expanded=True):
        render_candidate_pool_summary(pool)

    with st.expander("Skill Gap Analysis", expanded=True):
        render_skill_gap_analysis(skills)

    with st.expander("Candidate Score Distribution", expanded=True):
        render_candidate_distribution()

    with st.expander("Top Candidates", expanded=True):
        render_top_candidates(top)
        if st.checkbox("Generate AI insights for top candidates", key="gen_top_reasons"):
            if data_json and data_json != "[]":
                with st.spinner("Generating reasons..."):
                    reasons = _cached_top_reasons(data_json, cache_key)
                    rm = {r.get("name",""): r.get("reason","") for r in reasons}
                    for c in top: c["reason"] = rm.get(c["name"],"")
                render_top_candidates(top)

    with st.expander("AI Hiring Recommendation", expanded=False):
        if st.checkbox("Generate Hiring Recommendation", key="gen_hiring"):
            if data_json and data_json != "[]":
                with st.spinner("Analyzing candidate pool..."):
                    st.session_state.recruiter_insights["hiring_summary"] = _cached_hiring_summary(data_json, jd, cache_key)
        if "hiring_summary" in st.session_state.recruiter_insights:
            render_ai_hiring_recommendation(st.session_state.recruiter_insights["hiring_summary"])

    with st.expander("Candidate Comparison", expanded=False):
        render_candidate_comparison(cache_key)

    with st.expander("Interview Cutoff Suggestion", expanded=False):
        if st.checkbox("Suggest Interview Cutoff", key="gen_cutoff"):
            if data_json and data_json != "[]":
                with st.spinner("Analyzing cutoff..."):
                    st.session_state.recruiter_insights["interview_cutoff"] = _cached_cutoff(data_json, cache_key)
        if "interview_cutoff" in st.session_state.recruiter_insights:
            render_interview_cutoff(st.session_state.recruiter_insights["interview_cutoff"])

    with st.expander("Hiring Risks", expanded=False):
        if st.checkbox("Identify Hiring Risks", key="gen_risks"):
            if data_json and data_json != "[]":
                with st.spinner("Identifying risks..."):
                    st.session_state.recruiter_insights["hiring_risks"] = _cached_risks(data_json, cache_key)
        if "hiring_risks" in st.session_state.recruiter_insights:
            render_hiring_risks(st.session_state.recruiter_insights["hiring_risks"])

    with st.expander("Recruiter Actions", expanded=False):
        if st.checkbox("Generate Recommended Actions", key="gen_actions"):
            if data_json and data_json != "[]":
                with st.spinner("Generating actions..."):
                    st.session_state.recruiter_insights["recruiter_actions"] = _cached_actions(data_json, jd, cache_key)
        if "recruiter_actions" in st.session_state.recruiter_insights:
            render_recruiter_actions(st.session_state.recruiter_insights["recruiter_actions"])

    with st.expander("Export Report", expanded=False):
        if st.button("Generate PDF Report", type="primary", use_container_width=True, key="gen_pdf"):
            with st.spinner("Generating report..."):
                pdf = generate_recruiter_pdf(pool, skills, top,
                    st.session_state.recruiter_insights.get("hiring_risks",[]),
                    st.session_state.recruiter_insights.get("recruiter_actions",[]),
                    st.session_state.recruiter_insights.get("interview_cutoff",{}),
                    st.session_state.recruiter_insights.get("hiring_summary",{}))
                if pdf: st.session_state.recruiter_insights["pdf_report"] = pdf
        render_export_report(st.session_state.recruiter_insights.get("pdf_report"), "recruiter_report.pdf")

    _show_rule_based_warnings(pool)


# ---------------------------------------------------------------------------
# VIEWS
# ---------------------------------------------------------------------------

def set_page(page_name):
    st.session_state.current_page = page_name
    st.rerun()

def render_dashboard():
    st.markdown('<div class="page-title">Candidate Screening</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Automatically evaluate resumes, rank applicants, and export hiring recommendations.</div>', unsafe_allow_html=True)
    
    df = st.session_state.candidates_df
    results = st.session_state.evaluation_results
    
    total = len(df) if df is not None else 0
    parsed = len(st.session_state.resume_texts)
    shortlisted = sum(1 for r in results.values() if normalize_recommendation(r.get("recommendation", "")) == "Shortlist")
    avg_score = int(sum(r.get("score", 0) for r in results.values()) / len(results)) if results else 0
    
    cand_cap = "Ready for review" if total > 0 else "No data loaded"
    pars_cap = "Successfully processed" if parsed > 0 else "Pending extraction"
    sc_cap = "Excellent match quality" if avg_score >= 80 else "Average match quality" if avg_score > 0 else "No scores available"
    sh_cap = "Pipeline healthy" if shortlisted > 0 else "Awaiting recommendations"

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(html_metric_card("Candidates", str(total), cand_cap), unsafe_allow_html=True)
    with col2: st.markdown(html_metric_card("Resumes Parsed", str(parsed), pars_cap), unsafe_allow_html=True)
    with col3: st.markdown(html_metric_card("Average Score", str(avg_score), sc_cap), unsafe_allow_html=True)
    with col4: st.markdown(html_metric_card("Shortlist", str(shortlisted), sh_cap), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Quick Actions</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    
    with a1:
        st.markdown(f"""
        <div class="action-card">
            <div class="action-title">Upload Candidate Data</div>
            <div class="action-desc">Import candidate CSV and set job description.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Upload", key="btn_workspace", type="secondary", use_container_width=True): set_page("Upload")
            
    with a2:
        st.markdown(f"""
        <div class="action-card">
            <div class="action-title">Resume Processing</div>
            <div class="action-desc">Extract text and data from uploaded resumes.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Extract Resumes", key="btn_process", type="secondary", use_container_width=True): set_page("Upload")
            
    with a3:
        st.markdown(f"""
        <div class="action-card">
            <div class="action-title">AI Evaluation</div>
            <div class="action-desc">Run evaluation pipeline for all candidates.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Run Evaluation", key="btn_eval", type="secondary", use_container_width=True): set_page("Evaluate")


def render_upload():
    st.markdown('<div class="page-title">Upload Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Configure your requirements and import candidate data.</div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2, gap="large")
    
    with col_left:
        st.markdown('<div class="section-title" style="margin-top:0;">Job Description</div>', unsafe_allow_html=True)
        jd = st.text_area(
            "Job Description",
            value=st.session_state.job_description,
            height=280,
            label_visibility="collapsed",
            placeholder="Paste the job description here..."
        )
        st.session_state.job_description = jd
        st.caption(f"{len(jd)} characters")
        
    with col_right:
        st.markdown('<div class="section-title" style="margin-top:0;">Candidate Data</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            label_visibility="visible"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                df.index = range(len(df))
                st.session_state.candidates_df = df
            except Exception:
                st.error("Failed to parse CSV.")
                
        df = st.session_state.candidates_df
        if df is not None:
            st.markdown('<div class="table-container">', unsafe_allow_html=True)
            st.dataframe(df.head(4), use_container_width=True, height=180)
            
            resume_col = find_resume_column(df)
            if resume_col:
                st.markdown(f'<div class="upload-note" style="margin-top:12px;">Detected Resume Column: <span class="badge badge-neutral">{resume_col}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="upload-note" style="margin-top:12px;">No resume column detected.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state">No candidate data uploaded. Import a CSV file to begin.</div>', unsafe_allow_html=True)
                
    st.markdown('<div class="cc-divider"></div>', unsafe_allow_html=True)
    
    if df is not None:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="section-title" style="margin-top:0;">Process Resumes</div>', unsafe_allow_html=True)
            if st.session_state.resumes_processed:
                st.markdown(f'<div class="success-note">{len(st.session_state.resume_texts)} resumes parsed and ready.</div>', unsafe_allow_html=True)
            if st.button("Extract text from resumes", type="primary"):
                resume_col = find_resume_column(df)
                name_col = find_name_column(df)
                name_col = name_col if name_col else "Name"
                if not resume_col:
                    st.error("No valid resume column found.")
                    return
                
                resume_texts = {}
                total = len(df)
                progress_bar = st.progress(0, text="Initializing download...")
                
                for idx, row in df.iterrows():
                    name = str(row.get(name_col, f"Candidate {idx + 1}"))
                    progress_bar.progress((idx) / total, text=f"Processing {name} ({idx + 1}/{total})")
                    if pd.isna(row.get(resume_col)) or not str(row[resume_col]).strip(): continue
                    pdf_bytes = download_file(str(row[resume_col]).strip())
                    if not pdf_bytes: continue
                    text = extract_text_from_pdf(pdf_bytes)
                    if text and not text.startswith("[PDF"):
                        resume_texts[idx] = text
                        
                progress_bar.progress(1.0, text="Complete.")
                st.session_state.resume_texts = resume_texts
                st.session_state.resumes_processed = True
                st.rerun()

    
    st.markdown('<div class="cc-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Assessment Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle" style="margin-bottom:16px;">Upload external assessment scores (Coding, Aptitude, Logical, Communication).</div>', unsafe_allow_html=True)
    
    assess_file = st.file_uploader("Upload Assessment CSV", type=["csv"], key="assessment_uploader")
    candidates_for_assess = st.session_state.candidates_df
    
    if assess_file is not None:
        try:
            assess_df = pd.read_csv(assess_file)
            test_results, stats = process_assessment_csv(assess_df, candidates_for_assess)
            st.session_state.test_results = test_results
            st.session_state.assessment_stats = stats
            st.session_state.assessment_csv_uploaded = True
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.markdown(html_metric_card("Total Uploaded", str(stats["total"]), "Rows in CSV"), unsafe_allow_html=True)
            with col2: st.markdown(html_metric_card("Matched", str(stats["matched"]), "Linked to candidates"), unsafe_allow_html=True)
            with col3: st.markdown(html_metric_card("Unmatched", str(stats["unmatched"]), "No matching candidate"), unsafe_allow_html=True)
            with col4: st.markdown(html_metric_card("Detected Columns", ", ".join(stats["detected"]) if stats["detected"] else "None", "Assessment types found"), unsafe_allow_html=True)
        except Exception:
            st.error("Failed to parse assessment CSV.")
    elif st.session_state.assessment_csv_uploaded:
        s = st.session_state.assessment_stats
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(html_metric_card("Total Uploaded", str(s.get("total", 0)), "Rows in CSV"), unsafe_allow_html=True)
        with col2: st.markdown(html_metric_card("Matched", str(s.get("matched", 0)), "Linked to candidates"), unsafe_allow_html=True)
        with col3: st.markdown(html_metric_card("Unmatched", str(s.get("unmatched", 0)), "No matching candidate"), unsafe_allow_html=True)
        with col4: st.markdown(html_metric_card("Detected Columns", ", ".join(s.get("detected", [])) if s.get("detected") else "None", "Assessment types found"), unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">No assessment data uploaded.</div>', unsafe_allow_html=True)
    
def render_evaluate():
    st.markdown('<div class="page-title">Evaluation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Automated resume screening pipeline.</div>', unsafe_allow_html=True)
    
    df = st.session_state.candidates_df
    jd = st.session_state.job_description
    api_key = st.session_state.current_api_key
    processed = st.session_state.resumes_processed
    
    has_df = df is not None
    has_jd = bool(jd and len(jd) > 20)
    has_api = bool(api_key)
    
    def ind(ready: bool):
        return f'<div class="indicator {"ind-green" if ready else "ind-red"}"></div>'
    
    st.markdown(f"""
    <div class="ready-check-panel">
        <div class="section-title" style="margin-top:0;">Ready Checks</div>
        <div class="ready-check-item">{ind(has_df)} Candidate data loaded</div>
        <div class="ready-check-item">{ind(has_jd)} Job description configured</div>
        <div class="ready-check-item">{ind(has_api)} API key connected</div>
        <div class="ready-check-item">{ind(processed)} Resumes extracted</div>
    </div>
    """, unsafe_allow_html=True)
    
    can_eval = has_df and has_jd and has_api
    if not can_eval:
        st.markdown('<div class="empty-state">Please satisfy all ready checks to begin evaluation.</div>', unsafe_allow_html=True)
        return
        
    use_demo = st.checkbox("Enable Offline Demo Mode (Bypass API limits)", value=False)
    generate_explanations = st.checkbox("Generate AI score explanations", value=False)
    analyze_github = st.checkbox("Analyze GitHub profiles", value=False)

    st.markdown('<div class="cc-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Assessment Weights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle" style="margin-bottom:16px;">Configure weight distribution for Combined Recruiter Score.</div>', unsafe_allow_html=True)

    w_ai = st.slider("AI Evaluation", 0, 100, st.session_state.assessment_weights["ai"], key="w_ai")
    w_coding = st.slider("Coding Test", 0, 100, st.session_state.assessment_weights["coding"], key="w_coding")
    w_aptitude = st.slider("Aptitude Test", 0, 100, st.session_state.assessment_weights["aptitude"], key="w_aptitude")
    w_logical = st.slider("Logical Test", 0, 100, st.session_state.assessment_weights["logical"], key="w_logical")
    w_comm = st.slider("Communication", 0, 100, st.session_state.assessment_weights["communication"], key="w_communication")
    st.session_state.assessment_weights = {
        "ai": w_ai, "coding": w_coding, "aptitude": w_aptitude,
        "logical": w_logical, "communication": w_comm
    }
    total_w = sum(st.session_state.assessment_weights.values())
    if total_w != 100:
        st.caption(f"Total: {total_w}% (will be normalized to 100%)")

    if st.button("Run Evaluation", type="primary"):
        resume_col = find_resume_column(df)
        github_col = find_github_column(df)
        name_col = find_name_column(df)
        name_col = name_col if name_col else "Name"
        results = {}
        explanations = {}
        github_analysis = {}
        total = len(df)
        
        progress_bar = st.progress(0, text="Initializing evaluation sequence...")
        
        for idx, row in df.iterrows():
            try:
                name = str(row.get(name_col, f"Candidate {idx + 1}"))
                progress_bar.progress((idx) / total, text=f"Evaluating {name}...")
                
                resume_text = st.session_state.resume_texts.get(idx, "")
                candidate_info = {col: str(row[col]) for col in df.columns if col != resume_col and pd.notna(row[col]) and str(row[col]).strip()}
                
                if use_demo:
                    import random
                    time.sleep(0.5)
                    score = random.randint(40, 95)
                    results[int(idx)] = {
                        "score": score,
                        "recommendation": "Shortlist" if score > 80 else "Maybe" if score > 50 else "Reject",
                        "strengths": ["Python", "System Design"] if score > 70 else ["Communication"],
                        "weaknesses": ["Cloud Ops"] if score < 60 else ["None"],
                        "summary": "Simulated demo evaluation output."
                    }
                else:
                    results[int(idx)] = evaluate_candidate_with_gemini(candidate_info, resume_text, jd.strip(), api_key)
                    if generate_explanations:
                        progress_bar.progress((idx) / total, text=f"Generating AI explanation for {name}...")
                        explanation = generate_ai_explanation_with_gemini(candidate_info, resume_text, jd.strip(), results[int(idx)], api_key)
                        if explanation:
                            explanations[int(idx)] = explanation
                    if analyze_github:
                        progress_bar.progress((idx) / total, text=f"Analyzing GitHub profile for {name}...")
                        if github_col and pd.notna(row.get(github_col)) and str(row.get(github_col)).strip():
                            username = extract_github_username(str(row.get(github_col)).strip())
                            github_analysis[int(idx)] = build_github_analysis(username, api_key) if username else {"ok": False, "message": "GitHub analysis unavailable."}
                        else:
                            github_analysis[int(idx)] = {"ok": False, "message": "No GitHub profile available.", "missing": True}
                    time.sleep(4.2)
            except Exception:
                continue
                
        progress_bar.progress(1.0, text="Evaluation complete.")
        st.session_state.evaluation_results = results
        st.session_state.ai_explanations = explanations
        st.session_state.github_analysis = github_analysis
        st.session_state.evaluated = True
        set_page("Rankings")


def render_rankings():
    st.markdown('<div class="page-title">Candidate Rankings</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Review, filter, and export hiring recommendations.</div>', unsafe_allow_html=True)
    
    if not st.session_state.evaluated or not st.session_state.evaluation_results:
        st.markdown('<div class="empty-state">No evaluations complete. Run the evaluation engine first.</div>', unsafe_allow_html=True)
        return
        
    df = st.session_state.candidates_df
    results = st.session_state.evaluation_results
    name_col = find_name_column(df)
    name_col = name_col if name_col else "Name"

    ranked_data = []
    for idx, res in results.items():
        idx = int(idx)
        if idx < len(df):
            row = df.iloc[idx]
            ranked_data.append({
                "Rank": 0, "Name": str(row.get(name_col, f"Candidate {idx + 1}")),
                "Score": res.get("score", 0), "Recommendation": normalize_recommendation(res.get("recommendation", "")),
                "Summary": res.get("summary", ""),
                "Strengths": res.get("strengths", []), "Weaknesses": res.get("weaknesses", []),
                "_idx": idx
            })

    if not ranked_data: return
    
    ranked_df = pd.DataFrame(ranked_data)
    
    # Top Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(html_metric_card("Evaluated", str(len(results)), "Total processed"), unsafe_allow_html=True)
    with c2: st.markdown(html_metric_card("Shortlist", str(sum(1 for r in ranked_data if normalize_recommendation(r["Recommendation"]) == "Shortlist")), "High priority"), unsafe_allow_html=True)
    with c3: st.markdown(html_metric_card("Maybe", str(sum(1 for r in ranked_data if normalize_recommendation(r["Recommendation"]) == "Maybe")), "Secondary review"), unsafe_allow_html=True)
    with c4: st.markdown(html_metric_card("Reject", str(sum(1 for r in ranked_data if normalize_recommendation(r["Recommendation"]) == "Reject")), "Not a fit"), unsafe_allow_html=True)
    
    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
    
    # Toolbar
    all_recs = sorted(ranked_df["Recommendation"].unique().tolist())
    if "filter_rec" not in st.session_state: st.session_state.filter_rec = all_recs
    if "search_cand" not in st.session_state: st.session_state.search_cand = ""
    if "sort_by" not in st.session_state: st.session_state.sort_by = "Highest Score"

    t1, t2, t3, t4, t5 = st.columns([2, 2, 2, 1, 2])
    with t1:
        search_val = st.text_input("Search Candidate", value=st.session_state.search_cand, placeholder="Search by name...")
        st.session_state.search_cand = search_val
    with t2:
        filter_rec = st.multiselect("Filter", options=all_recs, default=st.session_state.filter_rec)
        st.session_state.filter_rec = filter_rec
    with t3:
        sort_val = st.selectbox("Sort", options=["Highest Score", "Lowest Score", "Highest Combined Score", "Lowest Combined Score", "Alphabetical"], index=["Highest Score", "Lowest Score", "Highest Combined Score", "Lowest Combined Score", "Alphabetical"].index(st.session_state.sort_by))
        st.session_state.sort_by = sort_val
    with t5:
        interview_filter = st.selectbox("Interview", options=["All Candidates", "Scheduled", "Pending"], index=["All Candidates", "Scheduled", "Pending"].index(st.session_state.interview_filter))
        st.session_state.interview_filter = interview_filter
    with t4:
        st.markdown('<div class="toolbar-spacer"></div>', unsafe_allow_html=True)
        export_df = ranked_df.copy()
        export_df["Strengths"] = export_df["Strengths"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        export_df["Weaknesses"] = export_df["Weaknesses"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        export_df["Combined Score"] = export_df["_idx"].apply(
            lambda idx: calculate_combined_score(
                st.session_state.evaluation_results.get(int(idx), {}).get("score", 0),
                st.session_state.test_results.get(int(idx)),
                st.session_state.assessment_weights
            ) or ""
        )
        export_df["Coding Score"] = export_df["_idx"].apply(lambda idx: (st.session_state.test_results.get(int(idx)) or {}).get("coding") or "")
        export_df["Aptitude Score"] = export_df["_idx"].apply(lambda idx: (st.session_state.test_results.get(int(idx)) or {}).get("aptitude") or "")
        export_df["Logical Score"] = export_df["_idx"].apply(lambda idx: (st.session_state.test_results.get(int(idx)) or {}).get("logical") or "")
        export_df["Communication Score"] = export_df["_idx"].apply(lambda idx: (st.session_state.test_results.get(int(idx)) or {}).get("communication") or "")
        export_df["Interview Status"] = export_df["_idx"].apply(lambda idx: (st.session_state.interview_schedule.get(int(idx)) or {}).get("status") or "")
        export_df["Interview Date"] = export_df["_idx"].apply(lambda idx: (st.session_state.interview_schedule.get(int(idx)) or {}).get("date") or "")
        export_df["Interview Time"] = export_df["_idx"].apply(lambda idx: (st.session_state.interview_schedule.get(int(idx)) or {}).get("time") or "")
        export_df["Meeting Mode"] = export_df["_idx"].apply(lambda idx: (st.session_state.interview_schedule.get(int(idx)) or {}).get("mode") or "")
        export_df["Meeting Link"] = export_df["_idx"].apply(lambda idx: (st.session_state.interview_schedule.get(int(idx)) or {}).get("meeting_link") or "")
        export_df["Interviewer"] = export_df["_idx"].apply(lambda idx: (st.session_state.interview_schedule.get(int(idx)) or {}).get("interviewer") or "")
        export_df = export_df[["Name", "Score", "Combined Score", "Recommendation", "Summary", "Strengths", "Weaknesses", "Coding Score", "Aptitude Score", "Logical Score", "Communication Score", "Interview Status", "Interview Date", "Interview Time", "Meeting Mode", "Meeting Link", "Interviewer"]]
        csv_buf = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("Export CSV", data=csv_buf, file_name="rankings.csv", mime="text/csv", type="secondary", use_container_width=True)

    # Filtering and Sorting Logic
    display_df = ranked_df
    
    if search_val:
        display_df = display_df[display_df["Name"].str.contains(search_val, case=False, na=False)]
        
    if filter_rec:
        display_df = display_df[display_df["Recommendation"].isin(filter_rec)]

    if st.session_state.interview_filter == "Scheduled":
        display_df = display_df[display_df["_idx"].apply(lambda idx: st.session_state.interview_schedule.get(int(idx)) is not None and st.session_state.interview_schedule[int(idx)].get("status") == "Scheduled")]
    elif st.session_state.interview_filter == "Pending":
        display_df = display_df[display_df["_idx"].apply(lambda idx: st.session_state.interview_schedule.get(int(idx)) is None or st.session_state.interview_schedule[int(idx)].get("status") != "Scheduled")]

    if sort_val == "Highest Score":
        display_df = display_df.sort_values("Score", ascending=False).reset_index(drop=True)
    elif sort_val == "Lowest Score":
        display_df = display_df.sort_values("Score", ascending=True).reset_index(drop=True)
    elif sort_val == "Highest Combined Score":
        display_df["_combined"] = display_df["_idx"].apply(lambda idx: calculate_combined_score(
            st.session_state.evaluation_results.get(int(idx), {}).get("score", 0),
            st.session_state.test_results.get(int(idx)),
            st.session_state.assessment_weights
        ) or -1)
        display_df = display_df.sort_values("_combined", ascending=False).drop(columns=["_combined"]).reset_index(drop=True)
    elif sort_val == "Lowest Combined Score":
        display_df["_combined"] = display_df["_idx"].apply(lambda idx: calculate_combined_score(
            st.session_state.evaluation_results.get(int(idx), {}).get("score", 0),
            st.session_state.test_results.get(int(idx)),
            st.session_state.assessment_weights
        ) or -1)
        display_df = display_df.sort_values("_combined", ascending=True).drop(columns=["_combined"]).reset_index(drop=True)
    elif sort_val == "Alphabetical":
        display_df = display_df.sort_values("Name", ascending=True).reset_index(drop=True)

    display_df["Rank"] = range(1, len(display_df) + 1)
    
    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
    
    if display_df.empty:
        st.markdown('<div class="empty-state">No candidates match the current filters.</div>', unsafe_allow_html=True)
        return
        
    for _, row in display_df.iterrows():
        score = int(row["Score"])
        score_badge_cls, progress_bg = get_score_css(score)
        rec = normalize_recommendation(str(row["Recommendation"]))
        rec_badge_cls = get_status_css(rec)
        explanation_html = html_ai_explanation_breakdown(st.session_state.ai_explanations.get(int(row["_idx"]), {}))
        github_data = st.session_state.github_analysis.get(int(row["_idx"])) if st.session_state.github_analysis else None
        github_html = render_github_insights(github_data) if st.session_state.github_analysis else ""
        
        test_data = st.session_state.test_results.get(int(row["_idx"]))
        combined_score = calculate_combined_score(score, test_data, st.session_state.assessment_weights) if test_data else None
        assessment_html = render_assessment_insights(test_data, combined_score)
        
        interview_data = st.session_state.interview_schedule.get(int(row["_idx"]))
        interview_html = render_schedule_status(interview_data)
        
        str_list = "".join(f"<li>{s}</li>" for s in row["Strengths"]) if isinstance(row["Strengths"], list) else f"<li>{row['Strengths']}</li>"
        wk_list = "".join(f"<li>{w}</li>" for w in row["Weaknesses"]) if isinstance(row["Weaknesses"], list) else f"<li>{row['Weaknesses']}</li>"
        
        st.markdown(f"""
        <div class="candidate-card">
            <div class="cc-header">
                <div>
                    <div class="cc-title">{row['Name']}</div>
                    <div class="cc-subtitle">AI Match Score</div>
                </div>
                <div class="cc-badges">
                    <div class="badge {score_badge_cls}">{score}%</div>
                    <div class="badge {rec_badge_cls}">{rec}</div>
                </div>
            </div>
            
            <div class="cc-divider"></div>
            
            <div class="cc-summary">{row['Summary']}</div>
            {explanation_html}
            {github_html}
            {assessment_html}
            {interview_html}
            
            <div class="cc-divider"></div>
            
            <div class="cc-details">
                <div>
                    <div class="cc-section-title">Strengths</div>
                    <ul class="cc-list">{str_list}</ul>
                </div>
                <div>
                    <div class="cc-section-title">Weaknesses</div>
                    <ul class="cc-list">{wk_list}</ul>
                </div>
            </div>
            
            <div class="cc-divider"></div>
            
            <div class="cc-progress-container">
                <div class="cc-progress-bar {progress_bg}" style="width: {score}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.github_analysis:
            with st.expander("View GitHub Details"):
                render_github_details(github_data)


    render_interview_panel()
    
# ---------------------------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "current_page": "Dashboard", "candidates_df": None, "resume_texts": {},
        "evaluation_results": {}, "evaluated": False, "resumes_processed": False,
        "job_description": "", "current_api_key": GEMINI_API_KEY,
        "ai_explanations": {}, "github_analysis": {}, "test_results": {}, "assessment_weights": {"ai": 70, "coding": 20, "aptitude": 10, "logical": 0, "communication": 0}, "assessment_csv_uploaded": False, "assessment_stats": {}, "interview_schedule": {}, "interview_filter": "All Candidates",
        "recruiter_insights": {}, "copilot_cache_key": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def main():
    init_session_state()
    inject_css()

    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-logo"></div>
            <div>
                <div class="sidebar-title">AI Screener</div>
                <div class="sidebar-subtitle">Recruitment Workspace</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        pages = ["Dashboard", "Upload", "Evaluate", "Rankings", "Recruiter Copilot"]
        selection = st.pills("Navigation", pages, default=st.session_state.current_page, label_visibility="collapsed")
        if selection and selection != st.session_state.current_page:
            set_page(selection)
            
        st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">Configuration</div>', unsafe_allow_html=True)
        api_key = st.text_input("Gemini API Key", value=st.session_state.current_api_key, type="password", label_visibility="collapsed", placeholder="Gemini API Key")
        if api_key != st.session_state.current_api_key:
            st.session_state.current_api_key = api_key
            
        st.markdown(f"""
        <div class="sidebar-api-state sidebar-status">
            <div class="indicator {'ind-green' if st.session_state.current_api_key else 'ind-red'}"></div>
            {'API Connected' if st.session_state.current_api_key else 'API Disconnected'}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">Workspace Status</div>', unsafe_allow_html=True)
        df = st.session_state.candidates_df
        t_cand = len(df) if df is not None else 0
        t_res = len(st.session_state.resume_texts)
        t_eval = len(st.session_state.evaluation_results)
        
        st.markdown(f"""
        <div class="sidebar-row"><span>Candidates Loaded</span><span>{t_cand}</span></div>
        <div class="sidebar-row"><span>Resumes Parsed</span><span>{t_res}</span></div>
        <div class="sidebar-row"><span>Evaluations Complete</span><span>{t_eval}</span></div>
        """, unsafe_allow_html=True)

    page = st.session_state.current_page
    if page == "Dashboard": render_dashboard()
    elif page == "Upload": render_upload()
    elif page == "Evaluate": render_evaluate()
    elif page == "Rankings": render_rankings()
    elif page == "Recruiter Copilot": render_recruiter_copilot()

if __name__ == "__main__":
    main()

# Do not rewrite the application from scratch. Make targeted edits to the existing code. Preserve the existing structure wherever possible.
