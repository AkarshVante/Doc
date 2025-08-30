import os
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

# Load .env for local dev
load_dotenv()

# Constants / defaults
MODEL_ORDER = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
]
EMBEDDING_MODEL = "models/embedding-001"
FAISS_DIR = "faiss_index"

def init_config():
    """Configure API keys and the genai client. Call this once at startup."""
    GOOGLE_API_KEY = None
    try:
        GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        # running outside of Streamlit or secrets not set
        pass
    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise RuntimeError("Google API key not found. Set GOOGLE_API_KEY in Streamlit secrets or in your local .env")
    genai.configure(api_key=GOOGLE_API_KEY)
