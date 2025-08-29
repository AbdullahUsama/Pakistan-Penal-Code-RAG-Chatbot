"""
Database connection and client management module
"""
import streamlit as st
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
from config import WEAVIATE_URL, WEAVIATE_API_KEY, COHERE_APIKEY, SENTENCE_TRANSFORMER_MODEL


@st.cache_resource
def initialize_weaviate_client():
    """Initialize and cache the Weaviate client"""
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            headers={
                "X-Cohere-Api-Key": COHERE_APIKEY 
            }
        )
        return client
    except Exception as e:
        st.error(f"Failed to connect to Weaviate: {e}")
        return None


@st.cache_resource
def load_sentence_transformer():
    """Load and cache the sentence transformer model"""
    return SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
