import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Fallback to Streamlit secrets if running in Streamlit Cloud
try:
    import streamlit as st
    if not GROQ_API_KEY and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    if hasattr(st, "secrets") and "GROQ_MODEL" in st.secrets:
        GROQ_MODEL = st.secrets["GROQ_MODEL"]
except Exception:
    pass


def create_groq_client(api_key: str = None) -> Groq:
    key = api_key or GROQ_API_KEY
    if not key:
        return None
    return Groq(api_key=key)

