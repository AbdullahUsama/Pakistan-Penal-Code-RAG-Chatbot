def chunk_markdown_by_exact_names(markdown_file_path):
    """
    Reads a markdown file and chunks it based on a predefined list of exact chapter names.
    """
    
    chapter_names = [
        "# CHAPTER I",
        "# CHAPTER II",
        "# CHAPTER III",
        "# CHAPTER IV",
        "# CHAPTER V",
        "# CHAPTER VI",
        "# CHAPTER VII",
        "# CHAPTER VIII",
        "# CHAPTER IX",
        "# CHAPTER X",
        "# CHAPTER XI",
        "# CHAPTER XII",
        "# CHAPTER XIII",
        "# CHAPTER XIV",
        "# CHAPTER XV",
        "# CHAPTER XVI",
        "# CHAPTER XVII",
        "# CHAPTER XVIII",
        "# CHAPTER XIX",
        "# CHAPTER XX",
        "# CHAPTER XXI",
        "# CHAPTER XXII",
        "# CHAPTER XXIII"
    ]

    try:
        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{markdown_file_path}' was not found.")
        return []

    chunks = []
    start_index = 0
    
    for i in range(len(chapter_names)):
        chapter_title = chapter_names[i]
        
        # Find the starting position of the current chapter
        try:
            current_chapter_start = markdown_text.index(chapter_title, start_index)
        except ValueError:
            # If the chapter name isn't found, something is wrong with the document.
            print(f"Warning: Chapter '{chapter_title}' not found. Skipping...")
            continue
            
        # Find the starting position of the next chapter to determine the end of the current chunk
        if i + 1 < len(chapter_names):
            next_chapter_title = chapter_names[i+1]
            try:
                next_chapter_start = markdown_text.index(next_chapter_title, current_chapter_start + len(chapter_title))
            except ValueError:
                # If this is the last chapter, the rest of the document is the content
                next_chapter_start = len(markdown_text)
        else:
            # This is the last chapter, so its content runs to the end of the document
            next_chapter_start = len(markdown_text)

        # Extract the content for the current chapter
        content = markdown_text[current_chapter_start + len(chapter_title):next_chapter_start].strip()
        
        chunks.append({
            "chapter_title": chapter_title,
            "content": content
        })
        
        # Update the start index for the next search
        start_index = next_chapter_start

    return chunks

# --- Example Usage ---
# Replace 'ppc.md' with the actual path to your markdown file.
file_path = 'article-embedder/ppc.md'
chapter_chunks = chunk_markdown_by_exact_names(file_path)

if chapter_chunks:
    print(f"Successfully chunked the document into {len(chapter_chunks)} chapters.")
    print("\n--- List of all Chapter Titles ---")
    
    for chunk in chapter_chunks:
        print(chunk['chapter_title'])
else:
    print("No chapters were found or the file could not be processed.")