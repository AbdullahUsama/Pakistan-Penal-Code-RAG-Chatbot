# Project Structure Documentation

## Overview
The Pakistan Penal Code RAG Chatbot has been modularized for better maintainability, readability, and organization. Below is the breakdown of each module and its responsibilities.

## File Structure

```
ppc-rag/
├── streamlit_app.py          # Main application entry point
├── config.py                 # Configuration and environment variables
├── database.py               # Database connections and client management
├── query_processing.py       # Query classification and processing
├── search_engine.py          # RAG search and response generation
├── ui_components.py          # UI components and styling
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not in repo)
└── README.md                 # Project documentation
```

## Module Descriptions

### 1. `config.py`
**Purpose**: Centralized configuration management
- Environment variable loading
- Configuration constants
- Model parameters
- Search parameters

### 2. `database.py`
**Purpose**: Database connection and client management
- Weaviate client initialization
- Sentence transformer model loading
- Caching mechanisms using Streamlit's `@st.cache_resource`

### 3. `query_processing.py`
**Purpose**: Query analysis and preprocessing
- Query classification (Legal vs General)
- General query response handling
- Query optimization for RAG system
- Gemini AI integration for classification

### 4. `search_engine.py`
**Purpose**: Core RAG functionality
- Semantic reranking of retrieved documents
- Vector database search
- Response generation using Gemini AI
- Context chunking and processing

### 5. `ui_components.py`
**Purpose**: User interface components and styling
- Custom CSS styling
- Header rendering
- Sidebar components
- Chat interface
- Debug information display

### 6. `streamlit_app.py`
**Purpose**: Main application orchestrator
- Application initialization
- User input processing
- Component coordination
- Session state management

## Benefits of Modularization

### 1. **Maintainability**
- Each module has a single responsibility
- Easier to locate and fix bugs
- Simplified testing and debugging

### 2. **Reusability**
- Components can be easily reused across different parts of the application
- Easier to extend functionality

### 3. **Readability**
- Code is organized logically
- Reduced complexity in individual files
- Better documentation and understanding

### 4. **Scalability**
- Easy to add new features
- Simple to modify individual components
- Better separation of concerns

### 5. **Collaboration**
- Multiple developers can work on different modules
- Reduced merge conflicts
- Clear ownership of code sections

## Usage

To run the application:

```bash
streamlit run streamlit_app.py
```

The main application will import and coordinate all the modular components automatically.

## Environment Variables

Make sure to create a `.env` file with the following variables:

```env
WEAVIATE_URL=your_weaviate_url
WEAVIATE_API_KEY=your_weaviate_api_key
COHERE_APIKEY=your_cohere_api_key
GEMINI_API_KEY=your_gemini_api_key
COLLECTION_NAME=your_collection_name
```

## Future Enhancements

With this modular structure, you can easily:
- Add new search algorithms in `search_engine.py`
- Implement new UI themes in `ui_components.py`
- Add new query types in `query_processing.py`
- Integrate new databases in `database.py`
- Modify configurations in `config.py`
