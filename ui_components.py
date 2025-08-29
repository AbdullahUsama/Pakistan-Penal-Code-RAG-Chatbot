"""
UI components and styling for the Streamlit app
"""
import streamlit as st
from contextlib import contextmanager


@contextmanager
def sidebar_spinner(text):
    """Context manager for displaying spinners in the sidebar"""
    with st.sidebar:
        with st.spinner(text):
            yield


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 2rem 1rem;
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .main-header p {
            font-size: 1.2rem;
            margin: 0;
            opacity: 0.9;
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
        
        /* Sidebar styling for spinners */
        .css-1d391kg {
            padding-top: 1rem;
        }
        
        /* Make sidebar spinners more visible */
        .sidebar .stSpinner > div {
            border-color: #2a5298 !important;
        }
        
        /* Mobile responsive styles */
        @media (max-width: 768px) {
            .main-header {
                padding: 1.5rem 0.5rem;
                margin-bottom: 1rem;
            }
            
            .main-header h1 {
                font-size: 1.8rem;
                line-height: 1.2;
            }
            
            .main-header p {
                font-size: 1rem;
                padding: 0 0.5rem;
            }
            
            /* Make chat input more mobile friendly */
            .stChatInputContainer {
                padding: 0 0.5rem;
            }
            
            /* Adjust title spacing on mobile */
            h1 {
                font-size: 1.5rem !important;
                margin-bottom: 1rem !important;
            }
            
            /* Better spacing for mobile */
            .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        
        @media (max-width: 480px) {
            .main-header {
                padding: 1rem 0.25rem;
            }
            
            .main-header h1 {
                font-size: 1.5rem;
            }
            
            .main-header p {
                font-size: 0.9rem;
            }
            
            h1 {
                font-size: 1.3rem !important;
            }
            
            /* Stack elements better on very small screens */
            .block-container {
                padding-left: 0.5rem;
                padding-right: 0.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the main header section"""
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Pakistan Penal Code AI Assistant</h1>
        <p>Get expert legal analysis and information about the Pakistan Penal Code</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with information and controls"""
    with st.sidebar:
        st.title("📚 About")
        st.info("""
        This AI assistant helps you understand the Pakistan Penal Code by:
        
        • Searching through all 23 chapters
        • Providing accurate legal information
        • Citing specific sections and chapters
        • Using advanced RAG technology
        """)
        
        # Status section for spinners
        st.markdown("---")
        st.subheader("🔄 Processing Status")
        st.caption("AI processing updates will appear here")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


def render_chat_interface():
    """Render the main chat interface"""
    st.markdown('<h1 style="text-align: center; margin-bottom: 1.5rem;">💬 Ask Your Legal Question</h1>', 
                unsafe_allow_html=True)
    
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


def render_debug_info(result):
    """Render debug information in an expander"""
    with st.expander("🔍 Debug Information"):
        st.write("**Query Type:** Legal (Using RAG)")
        st.write("**Optimized Query:**", result.get("optimized_query", "N/A"))
        st.write("**Retrieved Chunks:**")
        for i, chunk in enumerate(result.get("relevant_chunks", []), 1):
            st.write(f"**Chunk {i} ({chunk['chapter']})** - Score: {chunk['score']}")
            st.write(chunk['content'][:500] + "..." if len(chunk['content']) > 500 else chunk['content'])
            st.write("---")
