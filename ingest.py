import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# 1. Load Environment Variables
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# 2. Initialize Embedding Model (Runs locally, Free)
# This model converts text into a list of 384 numbers
print("Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_pdf_text(pdf_path):
    """Extract text from a PDF file"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text):
    """Split text into smaller chunks (so they fit in the AI's memory)"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Characters per chunk
        chunk_overlap=50      # Overlap to maintain context
    )
    return splitter.split_text(text)

def ingest_file(file_path):
    print(f"Processing {file_path}...")
    
    # Step A: Extract Text
    raw_text = get_pdf_text(file_path)
    
    # Step B: Chunk Text
    chunks = chunk_text(raw_text)
    print(f"Split into {len(chunks)} chunks.")
    
    # Step C: Embed and Upload to Pinecone
    vectors = []
    for i, chunk in enumerate(chunks):
        # Convert text to vector
        embedding = embed_model.encode(chunk).tolist()
        
        # Create metadata (so we know where the text came from)
        metadata = {"text": chunk, "source": file_path}
        
        # ID format: "filename_chunkIndex"
        file_id = f"{os.path.basename(file_path)}_{i}"
        
        vectors.append({
            "id": file_id,
            "values": embedding,
            "metadata": metadata
        })
    
    # Batch upload (Pinecone likes batches)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(batch)
        print(f"Uploaded batch {i // batch_size + 1}")

    print("Ingestion complete!")

# Run this block only if you run the script directly
# --- REPLACE THE BOTTOM SECTION WITH THIS ---

if __name__ == "__main__":
    # Check if directory exists
    if not os.path.exists("documents"):
        os.makedirs("documents")
        print("Created 'documents' folder. Please put a PDF file inside it.")
    else:
        # Find the first PDF in the directory
        files = [f for f in os.listdir("documents") if f.endswith('.pdf')]
        
        if files:
            pdf_name = files[0] # Take the first PDF found
            pdf_path = os.path.join("documents", pdf_name)
            print(f"Found file: {pdf_name}")
            ingest_file(pdf_path)
        else:
            print("Folder exists, but no PDF found. Please drag a PDF into the 'documents' folder.")