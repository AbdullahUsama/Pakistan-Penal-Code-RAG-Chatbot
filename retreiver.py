import weaviate
from weaviate.classes.init import Auth
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Load environment variables from .env file
load_dotenv()

def semantic_reranker(query, max_chunks=3, chunk_size=500, overlap=100):
    """
    Retrieve documents and rerank using semantic search
    
    Args:
        query: The search query
        max_chunks: Maximum number of chunks to return
        chunk_size: Size of each chunk in words
        overlap: Overlap between chunks in words
    
    Returns:
        List of semantically filtered chunks with similarity scores
    """
    
    # Initialize Weaviate client
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),                                    
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),  
        headers={
            "X-Cohere-Api-Key": os.getenv("COHERE_APIKEY")
        }       
    )
    
    collection = client.collections.get(os.getenv("COLLECTION_NAME"))
    
    # Initialize sentence transformer for semantic similarity
    sentence_model = SentenceTransformer('all-MiniLM-L12-v2')
    
    try:
        # Stage 1: Initial retrieval from Weaviate
        response = collection.query.hybrid(
            query=query,
            alpha=0.5,
            limit=5,  # Get more documents initially
            return_metadata=["score"]
        )
        
        all_chunks = []
        
        # Stage 2: Chunk the retrieved content
        for obj in response.objects:
            full_text = obj.properties.get('content', '')
            
            if full_text:
                # Split text into chunks with overlap
                words = full_text.split()
                
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = ' '.join(words[i:i + chunk_size])
                    if len(chunk.strip()) > 50:  # Only keep meaningful chunks
                        all_chunks.append(chunk)
                    
                    if i + chunk_size >= len(words):
                        break
        
        # Stage 3: Use semantic search to filter the best chunks
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
    
    finally:
        client.close()

# Usage
if __name__ == "__main__":
    query = "What is the punishment for abduction in Pakistan Penal Code?"
    
    filtered_results = semantic_reranker(query, max_chunks=3, chunk_size=400, overlap=50)
    
    for i, result in enumerate(filtered_results):
        print(f"Semantically Filtered Chunk {i+1} (Similarity: {result['similarity_score']:.3f}):")
        print(result['content'])
        print("-" * 50)