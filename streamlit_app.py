"""
Pakistan Penal Code AI Assistant - Main Streamlit Application

This is the main entry point for the PPC RAG Chatbot application.
The app has been modularized for better maintainability and organization.
"""
import streamlit as st
from database import initialize_weaviate_client
from query_processing import query_classifier, handle_general_query
from search_engine import search_and_generate_response
from ui_components import (
    apply_custom_css, 
    render_header, 
    render_sidebar, 
    render_chat_interface,
    render_debug_info,
    sidebar_spinner
)

def initialize_app():
    """Initialize the Streamlit application"""
    # Page configuration
    st.set_page_config(
        page_title="Pakistan Penal Code AI Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    apply_custom_css()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "client" not in st.session_state:
        st.session_state.client = None
    
    # Initialize client in the background
    if st.session_state.client is None:
        st.session_state.client = initialize_weaviate_client()


def process_user_input(user_question):
    """Process user input and generate appropriate response"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    # Check if client is available
    if not st.session_state.client:
        st.error("Database connection not available. Please refresh the page.")
        return
    
    # First, classify the query
    try:
        with sidebar_spinner("Analyzing your question..."):
            query_type = query_classifier(user_question)
        
        if query_type == "GENERAL":
            # Handle general conversational queries
            response = handle_general_query(user_question)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })
        
        elif query_type == "LEGAL":
            # Generate response using RAG system
            result = search_and_generate_response(st.session_state.client, user_question)
            
            if isinstance(result, dict):
                # Add assistant message to chat history
                assistant_message = {
                    "role": "assistant", 
                    "content": result["answer"],
                    "sources": result["sources"]
                }
                st.session_state.messages.append(assistant_message)
                
                # Show debug information
                render_debug_info(result)
                
            else:
                # Error occurred
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"❌ {result}"
                })
        
        else:
            # Fallback for unclear classification
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I'm not sure how to handle that query. Could you please ask a question about the Pakistan Penal Code or try rephrasing your question?"
            })
            
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"❌ An error occurred while processing your question: {e}"
        })


def main():
    """Main application function"""
    # Initialize the application
    initialize_app()
    
    # Render UI components
    render_header()
    render_sidebar()
    render_chat_interface()
    
    # Handle user input
    user_question = st.chat_input("Ask a question about the PPC...")
    
    if user_question:
        process_user_input(user_question)
        # Rerun to show the new messages
        st.rerun()


if __name__ == "__main__":
    main()
