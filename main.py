import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
import shutil

# --- Setup ---
load_dotenv()
app = FastAPI()

# Initialize Tools (But NOT the heavy model yet)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Global variable for the model (starts empty)
embed_model = None

def get_model():
    """
    Lazy Loader: This function downloads the model ONLY when we actually need it.
    This prevents the 'Port Scan Timeout' error on Render.
    """
    global embed_model
    if embed_model is None:
        print("Loading AI Model... (This might take a moment)")
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embed_model

# Request Model
class QueryRequest(BaseModel):
    question: str

# --- Helper Function: Semantic Search ---
def retrieve_context(query_text, top_k=3):
    """
    1. Turn query into vector
    2. Search Pinecone for similar vectors
    3. Return the text from those vectors
    """
    # Load the model NOW (if not already loaded)
    model = get_model()
    
    query_embedding = model.encode(query_text).tolist()
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    contexts = [match['metadata']['text'] for match in results['matches']]
    return "\n\n".join(contexts)

# --- Endpoint 1: Chat ---
@app.post("/chat")
async def chat(request: QueryRequest):
    user_query = request.question
    
    # Step 1: Find relevant data
    context = retrieve_context(user_query)
    
    

    # Step 2: Prepare Prompt for Groq
    system_prompt = """You are a specialized expert assistant. 
    You strictly answer questions based ONLY on the provided context below.
    If the answer is not in the context, say 'I don't know based on the documents provided.'
    Do not hallucinate or use outside knowledge."""
    
    user_prompt = f"""
    Context:
    {context}
    
    Question: 
    {user_query}
    """

    # Step 3: Call Groq
    # Note: We use the model you found earlier
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.3-70b-versatile", 
    )

    return {"answer": chat_completion.choices[0].message.content, "context_used": context}

# --- Endpoint 2: Upload Document ---
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Save file temporarily
    file_path = f"documents/{file.filename}"
    
    # Ensure directory exists
    if not os.path.exists("documents"):
        os.makedirs("documents")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run ingestion
    # We import here to avoid circular logic or startup delays
    from ingest import ingest_file
    ingest_file(file_path)
    
    return {"message": f"Successfully processed {file.filename}"}

# --- Root ---
@app.get("/")
def read_root():
    return {"status": "The Niche Knowledge Bot is Awake!"}