# Pakistan Penal Code RAG Chatbot ⚖️

🔗 **Live Demo**: [https://pakistan-penal-code-rag-chatbot.streamlit.app/](https://pakistan-penal-code-rag-chatbot.streamlit.app/)


![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28.0%2B-red.svg)
![Weaviate](https://img.shields.io/badge/weaviate-4.0.0%2B-green.svg)

A sophisticated **Retrieval-Augmented Generation (RAG)** chatbot designed to provide intelligent assistance for the Pakistan Penal Code. This system combines advanced document chunking, vector embeddings, and AI-powered question answering to deliver accurate legal information.

## Features

- **Intelligent Document Processing**: Hybrid chunking approach combining semantic and recursive strategies
- **Vector Database**: Powered by Weaviate for efficient similarity search
- **Advanced Embeddings**: Uses Cohere's multilingual embeddings for better understanding
- **AI-Powered Responses**: Integrated with Google's Gemini AI for natural language generation
- **Interactive Web Interface**: Built with Streamlit for user-friendly interactions
- **Real-time Search**: Fast and accurate retrieval of relevant legal sections

## Architecture

### Data Processing Pipeline
```
Pakistan Penal Code (Markdown) 
    ↓
Semantic Chunking (by Chapters & Sections)
    ↓
Recursive Chunking (size-based with overlap)
    ↓
Vector Embeddings (Cohere multilingual-v3.0)
    ↓
Weaviate Vector Database
```

### RAG System Flow

```
User Query 
    ↓
Query Optimization (Gemini 2.0 Flash)
    ├── Legal concept extraction
    ├── Chapter mapping
    └── Query enhancement
    ↓
Hybrid Search (Weaviate)
    ├── Semantic Search (90% weight)
    │   └── Cohere embed-multilingual-v3.0
    └── Keyword Search (10% weight)
    ↓
Initial Retrieval
    ├── Top 4 most relevant chunks
    └── Metadata scoring
    ↓
Semantic Reranking
    ├── Secondary chunking
    ├── Sentence-BERT similarity scoring
    └── Top 4 refined chunks selection
    ↓
Context Assembly & Prompt Engineering
    ├── Legal text formatting
    ├── Citation preparation
    ↓
AI Response Generation (Gemini 2.0 Flash)
    ├── Legal analysis
    ├── Section citations
    └── Structured response
    ↓
Response Post-processing
    ├── Source attribution
    ├── Chapter references
    └── Confidence indicators
```

### Advanced RAG Components

#### Query Optimization Pipeline
- **Legal Context Mapping**: Automatically maps queries to relevant PPC chapters
- **Concept Enhancement**: Expands legal terminology for better retrieval
- **Chapter-Specific Targeting**: Includes relevant chapter names in search

#### Hybrid Search Strategy
```python
# Search Configuration
Alpha = 0.6  # 60% semantic, 40% keyword
Limit = 4    # Initial retrieval count
Model = "embed-multilingual-v3.0"  # Cohere embeddings
```

#### 🧠 Response Generation
- **Model**: Gemini 2.0 Flash
- **Approach**: Context-aware legal analysis
- **Citations**: Automatic section and chapter referencing
- **Validation**: Response grounded in retrieved context only

## 📁 Project Structure

```
ppc-rag/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables (not tracked)
├── ppc.md                      # Pakistan Penal Code source document
├── streamlit_app.py            # Main Streamlit web application
├── weaviate_populate_v2.py     # Advanced data processing and upload
├── weaviate_populate.py        # Basic data processing script
├── bot.py                      # Core chatbot functionality
├── query_parser.py             # Query processing utilities
├── retreiver.py                # Document retrieval logic
└── simple_chunker.py           # Basic chunking utilities
```

## Advanced Chunking Strategy

Our system employs a **hybrid chunking approach** that combines semantic understanding with optimal chunk sizing:

### 1. **Semantic Chunking**
- **Chapter-level division**: Automatically identifies and separates chapters (I-XXIII)
- **Section-level extraction**: Uses regex patterns to identify numbered sections within chapters
- **Contextual preservation**: Maintains logical document structure and legal context

### 2. **Recursive Chunking**
- **Size-based splitting**: Further divides large sections into manageable chunks (300 words default)
- **Overlap strategy**: Implements 30-word overlap between chunks for context continuity
- **Adaptive processing**: Automatically handles sections of varying sizes

### 3. **Chunk Metadata**
Each chunk contains rich metadata for enhanced retrieval:
```python
{
    'chapter_title': '# CHAPTER I',
    'section_number': '302',
    'chunk_id': 'I-302-1',
    'content': 'Legal text content...',
    'word_count': 280,
    'chunk_type': 'section'  # or 'subsection'
}
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Weaviate Cloud account
- Cohere API key
- Google Gemini API key

### 1. Clone the Repository
```bash
git clone https://github.com/AbdullahUsama/Pakistan-Penal-Code-RAG-Chatbot.git
cd Pakistan-Penal-Code-RAG-Chatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```env
WEAVIATE_URL=your_weaviate_cluster_url
WEAVIATE_API_KEY=your_weaviate_api_key
COHERE_APIKEY=your_cohere_api_key
GEMINI_API_KEY=your_gemini_api_key
COLLECTION_NAME=PPC_2
```

### 4. Data Processing & Upload
```bash
python weaviate_populate_v2.py
```

### 5. Launch the Application
```bash
streamlit run streamlit_app.py
```

## 💾 Vector Database Schema

### Weaviate Collection Properties
- **chapter_title** (TEXT): Chapter identification
- **section_number** (TEXT): Section or subsection number
- **chunk_id** (TEXT): Unique chunk identifier
- **content** (TEXT): Actual legal text content
- **word_count** (INT): Number of words in chunk
- **chunk_type** (TEXT): Classification (section/subsection)

### Vectorization
- **Model**: Cohere `embed-multilingual-v3.0`
- **Dimensionality**: 1024-dimensional vectors
- **Language Support**: Optimized for English and Urdu legal text

## 🎯 Key Components

### `weaviate_populate_v2.py`
Advanced data processing script featuring:
- Hybrid chunking algorithm
- Batch upload optimization
- Error handling and logging
- Statistical analysis of chunks

### `streamlit_app.py`
Interactive web interface providing:
- Real-time query processing
- Formatted legal responses
- Source attribution
- User-friendly design

### `bot.py`
Core chatbot logic including:
- Query understanding
- Context retrieval
- Response generation
- Legal text formatting

## 📊 Performance Metrics

- **Chunk Size**: 300 words (configurable)
- **Overlap**: 30 words for context preservation
- **Processing Speed**: ~1000 chunks/minute
- **Retrieval Accuracy**: Vector similarity-based ranking
- **Response Time**: <2 seconds for typical queries

## 🔍 Usage Examples

### Basic Legal Query
```
Query: "What is the punishment for theft?"
Response: Relevant sections from Chapter XVII with penalties and legal provisions
```

### Section-Specific Search
```
Query: "Section 302 Pakistan Penal Code"
Response: Complete text of Section 302 (Murder) with context
```

### Conceptual Questions
```
Query: "Types of criminal conspiracy"
Response: Aggregated information from relevant sections about conspiracy
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Vector Database**: Weaviate Cloud
- **Embeddings**: Cohere API
- **LLM**: Google Gemini
- **Language**: Python 3.8+
- **Processing**: Custom hybrid chunking algorithm


## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Abdullah Usama**
- GitHub: [@AbdullahUsama](https://github.com/AbdullahUsama)
- Project Link: [Pakistan-Penal-Code-RAG-Chatbot](https://github.com/AbdullahUsama/Pakistan-Penal-Code-RAG-Chatbot)

---

## ⚠️ Disclaimer

This AI assistant is for informational purposes only and should not be considered as legal advice. Always consult with qualified legal professionals for legal matters.

---

⭐ **Star this repository if you find it useful!**

*Built with ❤️ for legal accessibility and AI innovation*
