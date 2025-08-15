# def extract_specific_headings(file_path):
#     """
#     Extracts specific chapter headings and Roman numeral headings
#     from a markdown file.

#     Args:
#         file_path (str): The path to the markdown file.

#     Returns:
#         list: A list of the specific headings found.
#     """
#     specific_headings = []
#     # Set of Roman numeral headings you want to include
#     roman_numerals = {}

#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 stripped_line = line.strip()
#                 # Check for "CHAPTER" headings
#                 if stripped_line.startswith('# CHAPTER'):
#                     specific_headings.append(stripped_line)
#                 # Check for the specified Roman numeral headings
#                 elif stripped_line in roman_numerals:
#                     specific_headings.append(stripped_line)
#     except FileNotFoundError:
#         print(f"Error: The file '{file_path}' was not found.")
#         return None
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None
    
#     return specific_headings

# # Call the function and store the returned list
# headings_list = extract_specific_headings("article-embedder/ppc.md")

# # Check if the list is not empty and then print the headings
# if headings_list:
#     for heading in headings_list:
#         print(heading)




##########################################################chunking
# def chunk_markdown_by_exact_names(markdown_file_path):
#     """
#     Reads a markdown file and chunks it based on a predefined list of exact chapter names.
#     """
    
#     chapter_names = [
#         "# CHAPTER I",
#         "# CHAPTER II",
#         "# CHAPTER III",
#         "# CHAPTER IV",
#         "# CHAPTER V",
#         "# CHAPTER VI",
#         "# CHAPTER VII",
#         "# CHAPTER VIII",
#         "# CHAPTER IX",
#         "# CHAPTER X",
#         "# CHAPTER XI",
#         "# CHAPTER XII",
#         "# CHAPTER XIII",
#         "# CHAPTER XIV",
#         "# CHAPTER XV",
#         "# CHAPTER XVI",
#         "# CHAPTER XVII",
#         "# CHAPTER XVIII",
#         "# CHAPTER XIX",
#         "# CHAPTER XX",
#         "# CHAPTER XXI",
#         "# CHAPTER XXII",
#         "# CHAPTER XXIII"
#     ]

#     try:
#         with open(markdown_file_path, 'r', encoding='utf-8') as f:
#             markdown_text = f.read()
#     except FileNotFoundError:
#         print(f"Error: The file '{markdown_file_path}' was not found.")
#         return []

#     chunks = []
#     start_index = 0
    
#     for i in range(len(chapter_names)):
#         chapter_title = chapter_names[i]
        
#         # Find the starting position of the current chapter
#         try:
#             current_chapter_start = markdown_text.index(chapter_title, start_index)
#         except ValueError:
#             # If the chapter name isn't found, something is wrong with the document.
#             print(f"Warning: Chapter '{chapter_title}' not found. Skipping...")
#             continue
            
#         # Find the starting position of the next chapter to determine the end of the current chunk
#         if i + 1 < len(chapter_names):
#             next_chapter_title = chapter_names[i+1]
#             try:
#                 next_chapter_start = markdown_text.index(next_chapter_title, current_chapter_start + len(chapter_title))
#             except ValueError:
#                 # If this is the last chapter, the rest of the document is the content
#                 next_chapter_start = len(markdown_text)
#         else:
#             # This is the last chapter, so its content runs to the end of the document
#             next_chapter_start = len(markdown_text)

#         # Extract the content for the current chapter
#         content = markdown_text[current_chapter_start + len(chapter_title):next_chapter_start].strip()
        
#         chunks.append({
#             "chapter_title": chapter_title,
#             "content": content
#         })
        
#         # Update the start index for the next search
#         start_index = next_chapter_start

#     return chunks

# # --- Example Usage ---
# # Replace 'ppc.md' with the actual path to your markdown file.
# file_path = 'article-embedder/ppc.md'
# chapter_chunks = chunk_markdown_by_exact_names(file_path)

# if chapter_chunks:
#     print(f"Successfully chunked the document into {len(chapter_chunks)} chapters.")
#     print("\n--- List of all Chapter Titles ---")
    
#     for chunk in chapter_chunks:
#         print(chunk['chapter_title'])
# else:
#     print("No chapters were found or the file could not be processed.")

#############################################################weaviate part
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
import os
import re
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util


# Load environment variables from .env file
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COHERE_APIKEY = os.getenv("COHERE_APIKEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

genai.configure(api_key=GEMINI_API_KEY)

def search_and_generate_response(client, query, collection_name=COLLECTION_NAME):
    """
    Search the vector database and generate a response using Gemini API
    """
    try:
        collection = client.collections.get(collection_name)
        
        # vector search
        # response = collection.query.near_text(
        #     query=query,
        #     limit=2,  # Get top 3 most relevant chunks
        #     return_metadata=["score"]
        # )

        print("DEBUG: Getting Rag optimzed query")
        rag_optimized_query = query_parser(query)
        print("DEBUG: RAG Optimized Query: ", rag_optimized_query)

        print("DEBUG: Getting Chapters from the DB")
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
        
        # Prepare context for Gemini
        context = ""
        for i, chunk in enumerate(relevant_chunks, 1):
            context += f"\n--- Relevant Section {i} ({chunk['chapter']}) ---\n"
            context += chunk['content'][:5000]  # Limit content length
            context += "\n"
        
        # Create prompt for Gemini

        print("DEBUG: Getting reranked chunks")
        reranked_context = semantic_reranker(query, relevant_chunks, max_chunks=2, chunk_size=700, overlap=200)
        print("DEBUG:CONTEXT:"+ context)
        print("DEBUG:RERANKED_CONTEXT" + str(reranked_context))

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
        """

        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-2.0-flash')
        gemini_response = model.generate_content(prompt)
        
        return {
            "answer": gemini_response.text,
            "sources": [chunk["chapter"] for chunk in relevant_chunks],
            "relevant_chunks": relevant_chunks
        }
        
    except Exception as e:
        return f"Error during search and generation: {e}"
def query_parser(query:str):
    user_query=query

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

    # Configure and generate content in one go
    # genai.configure(api_key=os.getenv("GEMINI_API_KEY")) 
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(PROMPT)

    return response.text



def semantic_reranker(query, relevant_chunks, max_chunks, chunk_size, overlap):
    """
    Rerank retrieved documents using semantic search
    
    Args:
        query: The search query
        relevant_chunks: List of already retrieved chunks from the database
        max_chunks: Maximum number of chunks to return
        chunk_size: Size of each chunk in words
        overlap: Overlap between chunks in words
    
    Returns:
        List of semantically filtered chunks with similarity scores
    """
    
    # Initialize sentence transformer for semantic similarity
    sentence_model = SentenceTransformer('all-MiniLM-L12-v2')
    
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

def interactive_chat(client):
    """
    Interactive chat interface using Gemini and Weaviate
    """
    print("\n" + "="*50)
    print("Pakistan Penal Code AI Assistant")
    print("Ask questions about the Pakistan Penal Code")
    print("Type 'quit' to exit")
    print("="*50 + "\n")
    
    while True:
        user_query = input("Your question: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_query:
            print("Please enter a valid question.")
            continue
        
        print("\nSearching and generating response...")
        result = search_and_generate_response(client, user_query)
        
        if isinstance(result, dict):
            print(f"\n--- Answer ---")
            print(result["answer"])
            print(f"\n--- Sources ---")
            print(f"Based on: {', '.join(result['sources'])}")
            print("\n" + "-"*50 + "\n")
        else:
            print(f"\nError: {result}\n")

if __name__ == "__main__":
    # --- Connect to Weaviate ---
    print("Connecting to Weaviate Cloud...")
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            headers={
                "X-Cohere-Api-Key": COHERE_APIKEY 
            }
        )
        print("Successfully connected to Weaviate.")
    except Exception as e:
        print(f"Failed to connect to Weaviate: {e}")
        exit()

    # Check if collection exists
    if not client.collections.exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' does not exist. Please run the data import first.")
        client.close()
        exit()

    # --- Start interactive chat ---
    try:
        interactive_chat(client)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        client.close()
        print("Connection to Weaviate closed.")

    # print(query_parser("what is punishment for abduction?"))