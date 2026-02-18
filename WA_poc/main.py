import os
import torch
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment variables
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable not set")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "")

# Initialize logging
logging.basicConfig(
    filename="api.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Initialize FastAPI app
app = FastAPI(
    title="Knowledge Base API",
    description="API for interacting with a knowledge base using MongoDB, ChromaDB, and Vanna AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# MongoDB setup with connection pooling
mongo_client = AsyncIOMotorClient(MONGO_URI, maxPoolSize=10, minPoolSize=2)
db = mongo_client["company_database"]
collection = db["company_info"]

# ChromaDB setup
chroma_settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PERSIST_DIR)
chroma_client = chromadb.Client(chroma_settings)
chroma_collection = chroma_client.create_collection("company_data")

# Sentence-BERT setup
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

# Function to generate embeddings
def get_embedding(text: str) -> List[float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

# Pydantic models
class CompanyRequest(BaseModel):
    company_id: str

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    results: List[str]
    search_time: float
    result_count: int

# MongoDB Data Extraction
async def get_data_from_mongo(company_id: str):
    company_data = await collection.find_one({"company_id": company_id})
    if not company_data:
        raise HTTPException(status_code=404, detail="Company data not found")
    return company_data

# Store Data in ChromaDB
async def store_data_in_chromadb(company_id: str):
    company_data = await get_data_from_mongo(company_id)
    company_name = company_data["company_name"]
    company_description = company_data["company_description"]
    
    embedding = get_embedding(company_description)
    
    chroma_collection.add(
        documents=[company_description],
        metadatas=[{"company_name": company_name}],
        embeddings=[embedding],
        ids=[company_id]
    )
    return company_name

# Query ChromaDB Knowledge Base
def query_chromadb(query: str):
    query_embedding = get_embedding(query)
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    return results['documents']

# Vanna AI integration (placeholder for actual Vanna AI implementation)
class VannaAI:
    def __init__(self):
        self.knowledge_base = chroma_collection
    
    def get_response(self, query: str):
        results = query_chromadb(query)
        if not results:
            return "No relevant information found."
        return "\n".join(results[0])

vanna_ai = VannaAI()

# FastAPI Routes
@app.post("/select_company", response_model=CompanyRequest)
async def select_company(company: CompanyRequest):
    try:
        company_name = await store_data_in_chromadb(company.company_id)
        return {"message": f"Company data for {company_name} has been successfully added to the knowledge base."}
    except Exception as e:
        logging.error(f"Error selecting company: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    start_time = datetime.now()
    try:
        response = vanna_ai.get_response(request.query)
        return {
            "query": request.query,
            "results": [response],
            "search_time": (datetime.now() - start_time).total_seconds(),
            "result_count": 1
        }
    except Exception as e:
        logging.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health Check
@app.get("/health")
async def health_check():
    try:
        await collection.count_documents({})  # Test MongoDB connection
        chroma_collection.peek()  # Test ChromaDB connection
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))

# Startup event
@app.on_event("startup")
async def startup_event():
    try:
        test_embedding = get_embedding("Test text for initialization check")
        if not test_embedding:
            raise Exception("Embedding model failed initialization check")
        logging.info("Application started successfully")
    except Exception as e:
        logging.error(f"Startup failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service initialization failed")

# Production configurations
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")