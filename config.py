"""
Configuration module for the PPC RAG Chatbot
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration constants
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COHERE_APIKEY = os.getenv("COHERE_APIKEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Model configuration
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L12-v2'
GEMINI_MODEL = 'gemini-2.0-flash'

# Search parameters
DEFAULT_SEARCH_LIMIT = 4
DEFAULT_ALPHA = 0.6
DEFAULT_MAX_CHUNKS = 4
DEFAULT_CHUNK_SIZE = 700
DEFAULT_OVERLAP = 200
