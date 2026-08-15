import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MODEL_FAST = "llama-3.1-8b-instant"

DATA_PATH = "data.json"

TOP_K = 5
FINAL_K = 5

CRAG_HIGH = 0.70  # >= this: answer directly
CRAG_LOW = 0.35   # < this: refuse
