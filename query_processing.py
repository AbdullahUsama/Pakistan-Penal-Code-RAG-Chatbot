"""
Query processing and classification module
"""
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


def query_classifier(query: str):
    """Classify if the query is related to PPC/Law or is a general conversational query"""
    
    CLASSIFICATION_PROMPT = f"""You are a query classification assistant. Your task is to determine if a user query is related to the Pakistan Penal Code (PPC) or legal matters, or if it's a general conversational query.

    **Classification Rules:**
    1. If the query is about law, legal matters, Pakistan Penal Code, sections, chapters, crimes, punishments, legal definitions, or any legal concepts - classify as "LEGAL"
    2. If the query is a general greeting, personal question, casual conversation, or unrelated to law - classify as "GENERAL"

    **Examples:**
    - "What is section 302?" → LEGAL
    - "Tell me about murder in PPC" → LEGAL
    - "What are the punishments for theft?" → LEGAL
    - "How are you?" → GENERAL
    - "Who are you?" → GENERAL
    - "What's the weather?" → GENERAL
    - "Hello" → GENERAL

    **User Query:** {query}

    **Instructions:** 
    Respond with only one word: either "LEGAL" or "GENERAL"
    """
    
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(CLASSIFICATION_PROMPT)
    return response.text.strip().upper()


def handle_general_query(query: str):
    """Handle general conversational queries"""
    query_lower = query.lower().strip()
    
    # Common greetings and responses
    if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
        return "Hello! I'm the Pakistan Penal Code AI Assistant. I'm here to help you with legal questions about the Pakistan Penal Code. How can I assist you today?"
    
    elif any(question in query_lower for question in ['how are you', 'how do you do', 'how\'s it going']):
        return "I'm doing well, thank you! I'm ready to help you with any questions about the Pakistan Penal Code. What would you like to know?"
    
    elif any(question in query_lower for question in ['who are you', 'what are you', 'tell me about yourself']):
        return """I'm an AI assistant specialized in the Pakistan Penal Code. I can help you:

• Find information about specific sections and chapters
• Explain legal definitions and concepts
• Understand punishments and legal procedures
• Navigate through all 23 chapters of the PPC

Feel free to ask me any legal questions related to the Pakistan Penal Code!"""
    
    elif any(question in query_lower for question in ['what can you do', 'what do you do', 'help']):
        return """I can help you with the Pakistan Penal Code by:

📚 **Searching through all 23 chapters**
⚖️ **Explaining legal sections and definitions**
🔍 **Finding relevant laws for specific situations**
📖 **Providing detailed legal analysis**
✅ **Citing specific sections and chapters**

Just ask me any question about Pakistani criminal law, and I'll provide you with accurate information from the PPC!"""
    
    elif 'thank' in query_lower:
        return "You're welcome! Feel free to ask me anything about the Pakistan Penal Code anytime."
    
    elif any(word in query_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
        return "Goodbye! Come back anytime you need help with the Pakistan Penal Code. Have a great day!"
    
    else:
        return """I'm designed to help specifically with Pakistan Penal Code related questions. 

For legal questions about the PPC, I can provide detailed analysis and cite relevant sections. 

Is there anything about Pakistani criminal law or the Pakistan Penal Code you'd like to know?"""


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

    If the Query has a specific section number mentioned, then keep the query very short and format
    it as ### Section 503. The three hashtags and a dot (.) at the end is a must.

    **Original Query:**
    {user_query}
    """

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(PROMPT)
    return response.text
