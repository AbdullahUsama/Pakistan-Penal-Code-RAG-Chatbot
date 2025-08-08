# script to get the chapters:
# CHAPTER I
# CHAPTER II
# CHAPTER III
# CHAPTER IV
# CHAPTER V
# CHAPTER VI
# CHAPTER VII
# CHAPTER VIII
# CHAPTER IX
# CHAPTER X
# CHAPTER XI
# CHAPTER XII
# CHAPTER XIII
# CHAPTER XIV
# CHAPTER XV
# CHAPTER XVI
# CHAPTER XVII
# CHAPTER XVIII
# CHAPTER XIX
# CHAPTER XX
# CHAPTER XXI
# CHAPTER XXII
# CHAPTER XXIII
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
        # Get the collection
        collection = client.collections.get(collection_name)
        
        # Perform vector search
        response = collection.query.near_text(
            query=query,
            limit=3,  # Get top 3 most relevant chunks
            return_metadata=["score"]
        )
        
        # Extract relevant content
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
        prompt = f"""Based on the following sections from the Pakistan Penal Code, please provide a comprehensive answer to the user's question.

User Question: {query}

Relevant Sections from Pakistan Penal Code:
{context}

Please provide a detailed answer based on the provided legal text. If the question cannot be fully answered from the given context, please mention that additional sections might be relevant."""

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

# ...existing code...

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