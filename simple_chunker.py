import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
import os
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COHERE_APIKEY = os.getenv("COHERE_APIKEY")

def simple_text_chunker(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Simple text chunking by word count with overlap
    
    Args:
        text: Input text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = ' '.join(words[start:end])
        chunks.append(chunk_text.strip())
        
        if end >= len(words):
            break
            
        start = end - overlap
    
    return chunks

def process_ppc_file(file_path: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Process PPC markdown file and create chunks
    
    Args:
        file_path: Path to the PPC markdown file
        chunk_size: Words per chunk
        overlap: Overlapping words between chunks
    
    Returns:
        List of chunk dictionaries
    """
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    
    # Remove markdown headers and clean text
    import re
    
    # Split by chapters to maintain some structure
    chapter_pattern = r'(# CHAPTER [IVX]+)'
    parts = re.split(chapter_pattern, content)
    
    all_chunks = []
    chunk_id = 1
    current_chapter = "UNKNOWN"
    
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
            
        # Check if this is a chapter header
        if part.startswith('# CHAPTER'):
            current_chapter = part
            continue
        
        # This is chapter content - chunk it
        text_chunks = simple_text_chunker(part, chunk_size, overlap)
        
        for chunk_text in text_chunks:
            if len(chunk_text.strip()) < 50:  # Skip very small chunks
                continue
                
            chunk = {
                'chunk_id': f"chunk_{chunk_id:04d}",
                'chapter_source': current_chapter,
                'content': chunk_text,
                'word_count': len(chunk_text.split()),
                'character_count': len(chunk_text)
            }
            all_chunks.append(chunk)
            chunk_id += 1
    
    return all_chunks

def create_collection(client, collection_name: str = "PPC-2"):
    """Create PPC-2 collection in Weaviate"""
    
    # Delete existing collection if it exists
    if client.collections.exists(collection_name):
        print(f"Deleting existing collection '{collection_name}'...")
        client.collections.delete(collection_name)
    
    # Create new collection
    print(f"Creating collection '{collection_name}'...")
    collection = client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.text2vec_cohere(
            model="embed-multilingual-v3.0"
        ),
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="chapter_source", data_type=DataType.TEXT),
            Property(name="content", data_type=DataType.TEXT),
            Property(name="word_count", data_type=DataType.INT),
            Property(name="character_count", data_type=DataType.INT),
        ]
    )
    
    return collection

def upload_to_weaviate(client, chunks: List[Dict], collection_name: str = "PPC-2"):
    """Upload chunks to Weaviate"""
    
    collection = client.collections.get(collection_name)
    
    print(f"Uploading {len(chunks)} chunks...")
    
    # Batch upload
    with collection.batch.dynamic() as batch:
        for chunk in chunks:
            batch.add_object(
                properties=chunk
            )
    
    # Verify
    total = collection.aggregate.over_all(total_count=True).total_count
    print(f"✅ Upload complete! Total objects: {total}")

def main():
    """Main execution function"""
    
    # ========== CONFIGURATION ==========
    FILE_PATH = 'article-embedder/ppc.md'  # Update this path
    COLLECTION_NAME = "PPC-2"
    CHUNK_SIZE = 300        # Adjust this: words per chunk
    OVERLAP = 30           # Adjust this: overlapping words
    # ===================================
    
    print("🔥 PPC Simple Chunker and Uploader")
    print(f"📄 File: {FILE_PATH}")
    print(f"📊 Chunk size: {CHUNK_SIZE} words")
    print(f"🔗 Overlap: {OVERLAP} words")
    print(f"🗂️  Collection: {COLLECTION_NAME}")
    print("-" * 50)
    
    # Connect to Weaviate
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            headers={"X-Cohere-Api-Key": COHERE_APIKEY}
        )
        print("✅ Connected to Weaviate")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    try:
        # Process file
        print("📝 Processing PPC file...")
        chunks = process_ppc_file(FILE_PATH, CHUNK_SIZE, OVERLAP)
        
        if not chunks:
            print("❌ No chunks created!")
            return
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Show stats
        word_counts = [c['word_count'] for c in chunks]
        print(f"📈 Stats: Avg {sum(word_counts)/len(word_counts):.1f} words, "
              f"Min {min(word_counts)}, Max {max(word_counts)}")
        
        # Preview first few chunks
        print("\n📋 Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"  {chunk['chunk_id']}: {chunk['content'][:80]}...")
        
        # Create collection and upload
        create_collection(client, COLLECTION_NAME)
        upload_to_weaviate(client, chunks, COLLECTION_NAME)
        
        print(f"\n🎉 Success! {len(chunks)} chunks uploaded to '{COLLECTION_NAME}'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()
