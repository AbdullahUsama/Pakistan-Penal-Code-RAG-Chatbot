import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
import os
import re
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COHERE_APIKEY = os.getenv("COHERE_APIKEY")

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return text

def chunk_text_by_size(text: str, chunk_size: int = 400, overlap: int = 100) -> List[str]:
    """
    Split text into chunks of approximately specified size with overlap
    
    Args:
        text: Input text to chunk
        chunk_size: Target number of words per chunk
        overlap: Number of words to overlap between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        
        if end >= len(words):
            break
            
        start = end - overlap
        if start <= 0:
            start = end
    
    return chunks

def extract_sections_from_chapter(content: str, chapter_title: str) -> List[Dict]:
    """
    Extract individual sections from a chapter content
    
    Args:
        content: Chapter content text
        chapter_title: Chapter title (e.g., "# CHAPTER I")
    
    Returns:
        List of section dictionaries
    """
    sections = []
    
    # Split by section numbers (e.g., "1.", "2.", "302.", etc.)
    section_pattern = r'\n(\d+\.)\s+'
    parts = re.split(section_pattern, content)
    
    if len(parts) <= 1:
        # No sections found, treat entire content as one section
        return [{
            'chapter_title': chapter_title,
            'section_number': 'General',
            'content': clean_text(content)
        }]
    
    # First part is usually intro text before first section
    if parts[0].strip():
        sections.append({
            'chapter_title': chapter_title,
            'section_number': 'Introduction',
            'content': clean_text(parts[0])
        })
    
    # Process section pairs (number, content)
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            section_num = parts[i].strip('.')
            section_content = parts[i + 1]
            
            sections.append({
                'chapter_title': chapter_title,
                'section_number': section_num,
                'content': clean_text(section_content)
            })
    
    return sections

def chunk_markdown_advanced(markdown_file_path: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Advanced chunking that creates smaller, more granular chunks
    
    Args:
        markdown_file_path: Path to the markdown file
        chunk_size: Target number of words per chunk
        overlap: Number of words to overlap between chunks
    
    Returns:
        List of chunk dictionaries
    """
    
    chapter_names = [
        "# CHAPTER I", "# CHAPTER II", "# CHAPTER III", "# CHAPTER IV", "# CHAPTER V",
        "# CHAPTER VI", "# CHAPTER VII", "# CHAPTER VIII", "# CHAPTER IX", "# CHAPTER X",
        "# CHAPTER XI", "# CHAPTER XII", "# CHAPTER XIII", "# CHAPTER XIV", "# CHAPTER XV",
        "# CHAPTER XVI", "# CHAPTER XVII", "# CHAPTER XVIII", "# CHAPTER XIX", "# CHAPTER XX",
        "# CHAPTER XXI", "# CHAPTER XXII", "# CHAPTER XXIII"
    ]

    try:
        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{markdown_file_path}' was not found.")
        return []

    all_chunks = []
    start_index = 0
    
    for i, chapter_title in enumerate(chapter_names):
        try:
            current_chapter_start = markdown_text.index(chapter_title, start_index)
        except ValueError:
            print(f"Warning: Chapter '{chapter_title}' not found. Skipping...")
            continue
            
        # Find the end of current chapter
        if i + 1 < len(chapter_names):
            next_chapter_title = chapter_names[i+1]
            try:
                next_chapter_start = markdown_text.index(next_chapter_title, current_chapter_start + len(chapter_title))
            except ValueError:
                next_chapter_start = len(markdown_text)
        else:
            next_chapter_start = len(markdown_text)

        # Extract chapter content
        chapter_content = markdown_text[current_chapter_start + len(chapter_title):next_chapter_start].strip()
        
        if not chapter_content:
            continue
        
        # Method 1: Extract by sections first
        sections = extract_sections_from_chapter(chapter_content, chapter_title)
        
        chunk_counter = 1
        for section in sections:
            section_content = section['content']
            
            if len(section_content.split()) <= chunk_size:
                # Section is small enough, use as single chunk
                chunk = {
                    'chapter_title': chapter_title,
                    'section_number': section['section_number'],
                    'chunk_id': f"{chapter_title.split()[-1]}-{section['section_number']}-{chunk_counter}",
                    'content': section_content,
                    'word_count': len(section_content.split()),
                    'chunk_type': 'section'
                }
                all_chunks.append(chunk)
                chunk_counter += 1
            else:
                # Section is too large, split into smaller chunks
                text_chunks = chunk_text_by_size(section_content, chunk_size, overlap)
                
                for j, text_chunk in enumerate(text_chunks):
                    chunk = {
                        'chapter_title': chapter_title,
                        'section_number': f"{section['section_number']}-part{j+1}",
                        'chunk_id': f"{chapter_title.split()[-1]}-{section['section_number']}-{chunk_counter}",
                        'content': text_chunk,
                        'word_count': len(text_chunk.split()),
                        'chunk_type': 'subsection'
                    }
                    all_chunks.append(chunk)
                    chunk_counter += 1
        
        start_index = next_chapter_start

    return all_chunks

def create_weaviate_collection(client, collection_name: str = "PPC-2"):
    """Create a new Weaviate collection for PPC chunks"""
    
    # Delete collection if it exists
    if client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' exists. Deleting...")
        client.collections.delete(collection_name)
    
    # Create new collection
    print(f"Creating collection '{collection_name}'...")
    collection = client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.text2vec_cohere(
            model="embed-multilingual-v3.0"
        ),
        properties=[
            Property(name="chapter_title", data_type=DataType.TEXT),
            Property(name="section_number", data_type=DataType.TEXT),
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="content", data_type=DataType.TEXT),
            Property(name="word_count", data_type=DataType.INT),
            Property(name="chunk_type", data_type=DataType.TEXT),
        ]
    )
    
    print(f"Collection '{collection_name}' created successfully!")
    return collection

def upload_chunks_to_weaviate(client, chunks: List[Dict], collection_name: str = "PPC-2", batch_size: int = 100):
    """Upload chunks to Weaviate collection"""
    
    collection = client.collections.get(collection_name)
    
    print(f"Uploading {len(chunks)} chunks to Weaviate...")
    
    # Upload in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        with collection.batch.dynamic() as batch_upload:
            for chunk in batch:
                batch_upload.add_object(
                    properties={
                        "chapter_title": chunk["chapter_title"],
                        "section_number": chunk["section_number"],
                        "chunk_id": chunk["chunk_id"],
                        "content": chunk["content"],
                        "word_count": chunk["word_count"],
                        "chunk_type": chunk["chunk_type"]
                    }
                )
        
        print(f"Uploaded batch {i//batch_size + 1} ({len(batch)} objects)")
    
    # Verify upload
    total_objects = collection.aggregate.over_all(total_count=True).total_count
    print(f"Total objects in collection: {total_objects}")

def main():
    """Main function to process and upload PPC data"""
    
    # Configuration
    MARKDOWN_FILE_PATH = 'ppc.md'  # Update this path as needed
    COLLECTION_NAME = "PPC_2"
    CHUNK_SIZE = 300  # Words per chunk (adjust as needed)
    OVERLAP = 50   # Word overlap between chunks
    
    print("="*60)
    print("Pakistan Penal Code - Advanced Chunking and Upload")
    print("="*60)
    
    # Connect to Weaviate
    print("Connecting to Weaviate...")
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            headers={
                "X-Cohere-Api-Key": COHERE_APIKEY 
            }
        )
        print("✅ Connected to Weaviate successfully!")
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {e}")
        return
    
    try:
        # Process markdown file
        print(f"\nProcessing markdown file: {MARKDOWN_FILE_PATH}")
        print(f"Chunk size: {CHUNK_SIZE} words, Overlap: {OVERLAP} words")
        
        chunks = chunk_markdown_advanced(
            MARKDOWN_FILE_PATH, 
            chunk_size=CHUNK_SIZE, 
            overlap=OVERLAP
        )
        
        if not chunks:
            print("❌ No chunks created. Check your file path and content.")
            return
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Show statistics
        word_counts = [chunk['word_count'] for chunk in chunks]
        print(f"📊 Chunk statistics:")
        print(f"   - Average words per chunk: {sum(word_counts) / len(word_counts):.1f}")
        print(f"   - Min words: {min(word_counts)}")
        print(f"   - Max words: {max(word_counts)}")
        
        # Show sample chunks
        print(f"\n📋 Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"   {i+1}. {chunk['chunk_id']} ({chunk['word_count']} words)")
            print(f"      Content preview: {chunk['content'][:100]}...")
        
        # Create collection
        create_weaviate_collection(client, COLLECTION_NAME)
        
        # Upload chunks
        upload_chunks_to_weaviate(client, chunks, COLLECTION_NAME)
        
        print(f"\n✅ Successfully uploaded {len(chunks)} chunks to '{COLLECTION_NAME}'!")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
    
    finally:
        client.close()
        print("\n🔌 Disconnected from Weaviate")

if __name__ == "__main__":
    main()
