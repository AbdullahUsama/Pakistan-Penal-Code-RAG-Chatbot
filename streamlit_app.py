import streamlit as st
import weaviate
from weaviate.classes.init import Auth
import os
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import time

# Load environment variables from .env file
load_dotenv()

# Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COHERE_APIKEY = os.getenv("COHERE_APIKEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Page configuration
st.set_page_config(
    page_title="Pakistan Penal Code AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #2a5298;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #1e3c72;
    }
</style>
""", unsafe_allow_html=True)

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
    return SentenceTransformer('all-MiniLM-L12-v2')

def query_parser(query: str):
    """Parse and optimize the user query for better RAG performance"""
    user_query = query

    PROMPT = f"""You are a query optimization assistant for a Retrieval-Augmented Generation (RAG) system.
    Your task is to rephrase a given user query into a highly effective query for a vector
    database containing sections of the Pakistan Penal Code. The optimized query should remove
    unnecessary conversational elements and focus on the core legal concepts, keywords,
    and section numbers relevant to the user's intent. When a query relates to a specific
    topic, include the relevant chapter numbers from the list below to narrow the search. Make sure 
    you add the complete chapter name like "CHAPTER V: OF ABETMENT" in the optimized prompt.

    **Pakistan Penal Code Chapters:**
    - CHAPTER I: INTRODUCTION
    - CHAPTER II: GENERAL EXPLANATIONS
    - CHAPTER III: OF PUNISHMENTS
    - CHAPTER IV: GENERAL EXCEPTIONS
    - CHAPTER V: OF ABETMENT
    - CHAPTER VI: OF OFFENCES AGAINST THE STATE
    - CHAPTER VII: OF OFFENCES RELATING TO THE ARMY, NAVY AND AIR FORCE
    - CHAPTER VIII: OF OFFENCES AGAINST THE PUBLIC TRANQUILLITY
    - CHAPTER IX: OF OFFENCES BY OR RELATING TO PUBLIC SERVANTS
    - CHAPTER X: OF CONTEMPTS OF THE LAWFUL AUTHORITY OF PUBLIC SERVANTS
    - CHAPTER XI: OF FALSE EVIDENCE AND OFFENCES AGAINST PUBLIC JUSTICE
    - CHAPTER XII: OF OFFENCES RELATING TO COIN AND GOVERNMENT STAMPS
    - CHAPTER XIII: OF OFFENCES RELATING TO WEIGHTS AND MEASURES
    - CHAPTER XIV: OF OFFENCES AFFECTING THE PUBLIC HEALTH, SAFETY, CONVENIENCE, DECENCY AND MORALS
    - CHAPTER XV: OF OFFENCES RELATING TO RELIGION
    - CHAPTER XVI: OF OFFENCES AFFECTING THE HUMAN BODY
    - CHAPTER XVII: OF OFFENCES AGAINST PROPERTY
    - CHAPTER XVIII: OF OFFENCES RELATING TO DOCUMENTS AND TO TRADE OR PROPERTY MARKS
    - CHAPTER XIX: OF THE CRIMINAL BREACH OF CONTRACTS OF SERVICE
    - CHAPTER XX: OF OFFENCES RELATING TO MARRIAGE
    - CHAPTER XXI: OF DEFAMATION
    - CHAPTER XXII: OF CRIMINAL INTIMIDATION, INSULT AND ANNOYANCE
    - CHAPTER XXIII: OF ATTEMPTS TO COMMIT OFFENCES

    **Original Query:**
    {user_query}
    """

    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(PROMPT)
    return response.text

def semantic_reranker(query, relevant_chunks, max_chunks, chunk_size, overlap):
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
        with st.spinner("Optimizing query..."):
            rag_optimized_query = query_parser(query)
        
        # Search the database
        with st.spinner("Searching Pakistan Penal Code..."):
            response = collection.query.hybrid(
                query=rag_optimized_query,
                alpha=0.9,
                limit=2,
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
        with st.spinner("Analyzing relevant sections..."):
            reranked_context = semantic_reranker(query, relevant_chunks, max_chunks=2, chunk_size=700, overlap=200)
        
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
        with st.spinner("Generating legal analysis..."):
            model = genai.GenerativeModel('gemini-2.0-flash')
            gemini_response = model.generate_content(prompt)
        
        return {
            "answer": gemini_response.text,
            "sources": [chunk["chapter"] for chunk in relevant_chunks],
            "relevant_chunks": relevant_chunks,
            "optimized_query": rag_optimized_query
        }
        
    except Exception as e:
        return f"Error during search and generation: {e}"

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Pakistan Penal Code AI Assistant</h1>
        <p>Get expert legal analysis and information about the Pakistan Penal Code</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "client" not in st.session_state:
        st.session_state.client = None
    
    # Sidebar
    with st.sidebar:
        st.title("📚 About")
        st.info("""
        This AI assistant helps you understand the Pakistan Penal Code by:
        
        • Searching through all 23 chapters
        • Providing accurate legal information
        • Citing specific sections and chapters
        • Using advanced RAG technology
        """)
        
        st.title("🔧 System Status")
        
        # Connection status
        if st.session_state.client is None:
            with st.spinner("Connecting to database..."):
                st.session_state.client = initialize_weaviate_client()
        
        if st.session_state.client:
            st.success("✅ Connected to Weaviate")
            
            # Check collection
            try:
                if st.session_state.client.collections.exists(COLLECTION_NAME):
                    st.success(f"✅ Collection '{COLLECTION_NAME}' ready")
                else:
                    st.error(f"❌ Collection '{COLLECTION_NAME}' not found")
            except:
                st.warning("⚠️ Unable to verify collection")
        else:
            st.error("❌ Database connection failed")
        
        st.title("💡 Example Questions")
        # example_questions = [
        #     "What is the punishment for murder?",
        #     "What constitutes theft under PPC?",
        #     "What are the offences against property?",
        #     "What is abetment according to PPC?",
        #     "What are the general exceptions in PPC?"
        # ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{question[:20]}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.title("💬 Ask Your Legal Question")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Show sources if available
                if "sources" in message:
                    st.info(f"📚 **Sources:** {', '.join(message['sources'])}")
    
    # Chat input
    user_question = st.chat_input("Ask a question about the Pakistan Penal Code...")
    
    if user_question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Check if client is available
        if not st.session_state.client:
            st.error("Database connection not available. Please refresh the page.")
            return
        
        # Generate response
        try:
            result = search_and_generate_response(st.session_state.client, user_question)
            
            if isinstance(result, dict):
                # Add assistant message to chat history
                assistant_message = {
                    "role": "assistant", 
                    "content": result["answer"],
                    "sources": result["sources"]
                }
                st.session_state.messages.append(assistant_message)
                
                # Show debug information in expander (moved here to avoid variable scope issues)
                with st.expander("🔍 Debug Information"):
                    st.write("**Optimized Query:**", result.get("optimized_query", "N/A"))
                    st.write("**Retrieved Chunks:**")
                    for i, chunk in enumerate(result.get("relevant_chunks", []), 1):
                        st.write(f"**Chunk {i} ({chunk['chapter']})** - Score: {chunk['score']}")
                        st.write(chunk['content'][:500] + "..." if len(chunk['content']) > 500 else chunk['content'])
                        st.write("---")
                
            else:
                # Error occurred
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"❌ {result}"
                })
                
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"❌ An error occurred: {e}"
            })
        
        # Rerun to show the new messages
        st.rerun()

if __name__ == "__main__":
    main()
