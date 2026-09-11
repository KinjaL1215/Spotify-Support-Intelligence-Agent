import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def create_groq_client(api_key: str = None) -> Groq:
    key = api_key or GROQ_API_KEY
    if not key:
        return None
    return Groq(api_key=key)
