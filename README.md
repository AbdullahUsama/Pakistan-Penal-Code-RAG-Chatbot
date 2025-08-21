# Pakistan Penal Code RAG Chatbot - Streamlit App

A Streamlit-based web application that provides an AI-powered assistant for querying the Pakistan Penal Code using Retrieval-Augmented Generation (RAG) technology.

## Features

- 🔍 **Intelligent Search**: Advanced semantic search through all 23 chapters of the Pakistan Penal Code
- 🤖 **AI-Powered Responses**: Uses Google Gemini for generating accurate legal analysis
- 📚 **Source Citations**: Provides specific chapter and section references
- 🎯 **Query Optimization**: Automatically optimizes user queries for better retrieval
- 🔄 **Semantic Reranking**: Re-ranks search results for maximum relevance
- 💬 **Interactive Chat**: User-friendly chat interface with conversation history
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Prerequisites

- Python 3.8 or higher
- Weaviate Cloud account (or local Weaviate instance)
- Google Gemini API key
- Cohere API key (for Weaviate)

## Setup

### 1. Environment Variables

Create a `.env` file in the project root with the following variables:

```env
WEAVIATE_URL=your_weaviate_cluster_url
WEAVIATE_API_KEY=your_weaviate_api_key
COHERE_APIKEY=your_cohere_api_key
GEMINI_API_KEY=your_gemini_api_key
COLLECTION_NAME=your_collection_name
```

### 2. Quick Start

#### Option A: Using the Batch File (Windows)
```bash
# Double-click run_app.bat or run in Command Prompt
run_app.bat
```

#### Option B: Using PowerShell
```powershell
# Run in PowerShell
.\run_app.ps1
```

#### Option C: Manual Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

### 3. Access the Application

After running the app, open your browser and navigate to:
```
http://localhost:8501
```

## Usage

1. **Ask Questions**: Type your legal questions in the chat input at the bottom
2. **Example Questions**: Use the sidebar example questions to get started
3. **View Sources**: Each response includes citations to specific chapters
4. **Clear History**: Use the "Clear Chat History" button to start fresh
5. **Debug Info**: Expand the debug section to see query optimization details

### Example Questions

- "What is the punishment for murder?"
- "What constitutes theft under PPC?"
- "What are the offences against property?"
- "What is abetment according to PPC?"
- "What are the general exceptions in PPC?"

## File Structure

```
ppc-rag/
├── streamlit_app.py      # Main Streamlit application
├── bot.py               # Original command-line version
├── requirements.txt     # Python dependencies
├── run_app.bat         # Windows batch launcher
├── run_app.ps1         # PowerShell launcher
├── .env                # Environment variables (create this)
└── README.md           # This file
```

## Technical Details

### Architecture

1. **Query Processing**: User queries are optimized using Gemini API
2. **Vector Search**: Hybrid search in Weaviate database
3. **Semantic Reranking**: Results are reranked using sentence transformers
4. **Response Generation**: Final response generated using Gemini API
5. **UI**: Clean, responsive Streamlit interface

### Key Components

- **Weaviate**: Vector database for storing PPC chapters
- **Google Gemini**: LLM for query optimization and response generation
- **Sentence Transformers**: For semantic similarity and reranking
- **Streamlit**: Web framework for the user interface

## Troubleshooting

### Common Issues

1. **Connection Error**: Check your `.env` file and API keys
2. **Collection Not Found**: Ensure your Weaviate collection exists and is populated
3. **Import Errors**: Run `pip install -r requirements.txt` to install dependencies
4. **Slow Responses**: This is normal for the first query as models are loaded

### Performance Tips

- The first query may take longer due to model loading
- Subsequent queries will be faster due to caching
- Use specific questions for better results
- Include relevant legal terms in your queries

## Development

To modify or extend the application:

1. **Core Logic**: Edit `streamlit_app.py` for UI changes
2. **RAG Functions**: Modify the search and generation functions
3. **Styling**: Update the CSS in the `st.markdown` sections
4. **Configuration**: Adjust parameters in the function calls

## License

This project is for educational and research purposes. Please ensure compliance with relevant legal and ethical guidelines when using this tool for legal information.

## Disclaimer

This AI assistant is for informational purposes only and should not be considered as legal advice. Always consult with qualified legal professionals for legal matters.
