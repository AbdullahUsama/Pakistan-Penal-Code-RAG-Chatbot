"""
Search and retrieval module for the RAG system
"""
import streamlit as st
import google.generativeai as genai
from sentence_transformers import util
from database import load_sentence_transformer
from query_processing import query_parser
from ui_components import sidebar_spinner
from config import (
    COLLECTION_NAME, 
    GEMINI_MODEL,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_ALPHA,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP
)


def semantic_reranker(query, relevant_chunks, max_chunks=DEFAULT_MAX_CHUNKS, 
                     chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """Rerank retrieved documents using semantic search"""
    sentence_model = load_sentence_transformer()
    
    all_chunks = []
    
    # Stage 1: Chunk the retrieved content
    for chunk in relevant_chunks:
        full_text = chunk.get('content', '')
        
        if full_text:
            # Split text into chunks with overlap
            words = full_text.split()
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_text = ' '.join(words[i:i + chunk_size])
                if len(chunk_text.strip()) > 30:  # Only keep meaningful chunks
                    all_chunks.append(chunk_text)
                
                if i + chunk_size >= len(words):
                    break
    
    # Stage 2: Use semantic search to filter the best chunks
    if all_chunks:
        # Get embeddings for query and chunks
        query_embedding = sentence_model.encode(query)
        chunk_embeddings = sentence_model.encode(all_chunks)
        
        # Calculate cosine similarities
        similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = similarities.argsort(descending=True)[:max_chunks]
        
        # Return chunks with their similarity scores
        filtered_chunks = []
        for idx in top_indices:
            filtered_chunks.append({
                'content': all_chunks[idx],
                'similarity_score': float(similarities[idx])
            })
        
        return filtered_chunks
    
    return []


def search_and_generate_response(client, query, collection_name=COLLECTION_NAME):
    """Search the vector database and generate a response using Gemini API"""
    try:
        collection = client.collections.get(collection_name)
        
        # Get optimized query
        with sidebar_spinner("Optimizing query..."):
            rag_optimized_query = query_parser(query)
        
        # Search the database
        with sidebar_spinner("Searching Pakistan Penal Code..."):
            response = collection.query.hybrid(
                query=rag_optimized_query,
                alpha=DEFAULT_ALPHA,
                limit=DEFAULT_SEARCH_LIMIT,
                return_metadata=["score"]
            )
        
        relevant_chunks = []
        for obj in response.objects:
            relevant_chunks.append({
                "chapter": obj.properties["chapter_title"],
                "content": obj.properties["content"],
                "score": obj.metadata.score if obj.metadata else "N/A"
            })
        
        if not relevant_chunks:
            return "No relevant information found in the Pakistan Penal Code."
        
        # Semantic reranking
        with sidebar_spinner("Analyzing relevant sections..."):
            reranked_context = semantic_reranker(query, relevant_chunks)
        
        # Create prompt for Gemini
        prompt = f"""You are a legal expert specializing in the Pakistan Penal Code. Your task is to analyze the provided sections and answer the user's legal question.

        **User Question:**
        {query}

        **Relevant Legal Text:**
        {reranked_context}

        **Instructions:**
        1.  Formulate a detailed, clear, and comprehensive answer to the user's question using ONLY the provided legal text.
        2.  If the question cannot be fully addressed with the given information, state that the provided text is insufficient and that other sections of the Pakistan Penal Code may be relevant. Do NOT speculate or provide information from outside the given context.
        3.  Do not use conversational phrases like "Based on the provided context..." or "According to the sections you gave me...".
        4.  At the end of your response, list the specific sections and their corresponding chapter numbers from the **Relevant Legal Text** that support your answer. Use the format: `(Chapter [Number], Section [Number])`.

        **Example:**
        The punishment for murder is death or life imprisonment. (Chapter XVI, Section 302)
        If you dont have the chapter number do not write "No Chpater Number Available", instead just write the section number.
        """

        # Generate response using Gemini
        with sidebar_spinner("Generating legal analysis..."):
            model = genai.GenerativeModel(GEMINI_MODEL)
            gemini_response = model.generate_content(prompt)
        
        return {
            "answer": gemini_response.text,
            "sources": [chunk["chapter"] for chunk in relevant_chunks],
            "relevant_chunks": relevant_chunks,
            "optimized_query": rag_optimized_query
        }
        
    except Exception as e:
        return f"Error during search and generation: {e}"
