from fastapi import FastAPI, Query
from pymongo import MongoClient, errors as pymongo_errors
from typing import List, Union, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from bson import ObjectId
import os
import json
import logging
import re
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

if not MONGO_URI or not DB_NAME:
    logger.error("MONGO_URI and DB_NAME must be set in the .env file.")
    raise ValueError("MONGO_URI and DB_NAME must be set in .env")

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    client.admin.command("ping")
    logger.info(f"Connected successfully to MongoDB database: {DB_NAME}")
except pymongo_errors.ServerSelectionTimeoutError as err:
    logger.error(f"MongoDB connection failed: {err}")
    raise SystemExit("Could not connect to MongoDB.")

app = FastAPI(title="MongoDB Smart Search API", version="1.3")

class SearchResult(BaseModel):
    collection: str
    documents: List[dict]

class SummaryResult(BaseModel):
    collection: str
    matches: List[str]

class SearchResponse(BaseModel):
    company_name: Optional[str]
    results: Union[List[SearchResult], List[SummaryResult]]

ALLOWED_COLLECTIONS = [
    "companies", "groups", "counters",
    "leads", "lead-assignments", "lead-notes", "lead-rotations",
    "lead-visited-properties", "contacts", "contact-tags", "cold-leads",
    "campaigns", "campaign-templates", "campaign-payments",
    "email-track", "chat-cache", "chats", "chat-groups",
    "lands", "projects", "properties", "amenities", "bhk", "bhk-types",
    "broker-payments", "contract-payments", "contracts", "general-expenses",
    "brokers", "contractors", "attendenc",
    "documents-and-priorities", "miscellaneous-documents"
]

def clean_document(doc):
    if isinstance(doc, dict):
        return {k: clean_document(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [clean_document(item) for item in doc]
    elif isinstance(doc, ObjectId):
        return str(doc)
    else:
        return doc

def tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())

def score_document(doc_str: str, keywords: List[str]) -> int:
    doc_words = tokenize(doc_str)
    word_counts = Counter(doc_words)
    return sum(word_counts[k] for k in keywords if k in word_counts)

def get_company_name(company_id: str) -> Optional[str]:
    try:
        obj_id = ObjectId(company_id)
        company = db["companies"].find_one({"_id": obj_id})
        if company:
            return company.get("name", "Unknown Company")
    except Exception as e:
        logger.warning(f"Error fetching company name: {e}")
    return None

MAX_SAFE_DOCS = 20

def smart_search(db, query: str, company_id: str, max_docs_per_collection: int = 3):
    try:
        company_object_id = ObjectId(company_id)
    except Exception:
        logger.error(f"Invalid company_id format: {company_id}")
        return "error", [{"collection": "error", "documents": [{"error": "Invalid company_id"}]}]

    keywords = tokenize(query)
    all_collections = db.list_collection_names()
    collections = [col for col in all_collections if col in ALLOWED_COLLECTIONS]
    full_results = []
    summary_results = []
    total_matches = 0

    logger.info(f"Searching for keywords: {keywords} across {len(collections)} collections")

    for collection_name in collections:
        try:
            collection = db[collection_name]
            cursor = collection.find({"company": company_object_id})
            scored_docs = []
            matched_fields = {}

            for doc in cursor:
                try:
                    doc_str = json.dumps(doc, default=str).lower()
                    score = score_document(doc_str, keywords)
                    if score > 0:
                        scored_docs.append((score, clean_document(doc)))
                        total_matches += 1
                        for key, value in doc.items():
                            if isinstance(value, (str, int, float)):
                                if any(kw in str(value).lower() for kw in keywords):
                                    matched_fields[key] = True
                            elif isinstance(value, list):
                                for item in value:
                                    if any(kw in str(item).lower() for kw in keywords):
                                        matched_fields[key] = True
                except Exception:
                    continue

            if scored_docs:
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                top_docs = [doc for score, doc in scored_docs[:max_docs_per_collection]]
                full_results.append({
                    "collection": collection_name,
                    "documents": top_docs
                })
                summary_results.append({
                    "collection": collection_name,
                    "matches": list(matched_fields.keys())
                })

        except Exception as e:
            logger.warning(f"Failed in collection '{collection_name}': {e}")

    if total_matches > MAX_SAFE_DOCS:
        logger.info(f"Returning SUMMARY only (total matches = {total_matches})")
        return "summary", summary_results
    else:
        logger.info(f"Returning FULL results (total matches = {total_matches})")
        return "full", full_results

@app.get("/search", response_model=SearchResponse)
def search_documents(
    query: str = Query(..., description="Search text"),
    company_id: str = Query(..., description="MongoDB ObjectId of the company")
):
    try:
        logger.info(f"Search request: query='{query}' | company_id='{company_id}'")
        company_name = get_company_name(company_id)
        result_type, results = smart_search(db, query, company_id)
        return {"company_name": company_name, "results": results}
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"company_name": None, "results": [{"collection": "error", "documents": [{"error": str(e)}]}]}

@app.get("/status")
def check_status():
    try:
        client.admin.command("ping")
        return {"status": "MongoDB connection is healthy."}
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        return {"status": "MongoDB connection failed.", "error": str(e)}        
