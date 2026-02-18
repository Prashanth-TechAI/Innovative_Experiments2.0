import os
import json
import copy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime
from openai import OpenAI
from rapidfuzz import process

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# MongoDB setup
client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

# OpenAI setup
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI app
app = FastAPI(title="Leads AI Chatbot")

# Field mapping for leads collection
LEADS_FIELD_MAP = {
    "name": ["name", "lead name", "customer name"],
    "phone": ["phone", "mobile", "contact"],
    "leadStatus": ["lead status", "status", "stage", "converted", "ongoing"],
    "createdAt": ["created", "created at", "date added", "generated"],
    "minBudget": ["min budget", "minimum price"],
    "maxBudget": ["max budget", "maximum price"],
    "buyingTimeline": ["buying time", "purchase period"],
    "commissionPercent": ["commission", "broker cut"],
    "lastActivity": ["last active", "recent activity"],
}

# Reverse map: fuzzy match user input to correct MongoDB field
def resolve_field_name(user_field: str):
    choices = []
    for real_field, aliases in LEADS_FIELD_MAP.items():
        for alias in aliases:
            choices.append((alias.lower(), real_field))
    best_match, score, _ = process.extractOne(user_field.lower(), dict(choices).keys())
    return dict(choices)[best_match] if score > 60 else user_field

# Request/Response models
class QueryRequest(BaseModel):
    company_id: str
    question: str

class QueryResponse(BaseModel):
    answer: str
    query: dict

# Clean LLM output
def clean_llm_json(text):
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().endswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

# Interpret user question into Mongo-style query
async def interpret_question_with_field_mapping(question: str) -> dict:
    prompt = f"""
    You are a MongoDB expert. Given a user query, return a JSON object with:
    - collection: always "leads"
    - operation: one of ["find", "count", "sum", "avg"]
    - field: the field to aggregate (null for count/find)
    - filters: Mongo-style key-value filter object

    Important Notes:
    - The 'leadStatus' field is a string and can have values like:
    "Temporary Converted", "Interested", "Contacted", "Not Interested", etc.
    - Do NOT use boolean values unless the field is known to be boolean.
    - Always return valid JSON only.

    Question: "{question}"
    """

    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    cleaned = clean_llm_json(raw)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        raise ValueError(f"❌ Failed to parse LLM response as JSON:\n{cleaned}")

    # Fuzzy-resolve filter field names
    filters = parsed.get("filters", {})
    parsed["filters"] = {resolve_field_name(k): v for k, v in filters.items()}
    return parsed

# Run MongoDB query
async def execute_leads_query(parsed: dict, company_id: str):
    collection = db["leads"]
    filters = parsed.get("filters", {})
    filters["company"] = ObjectId(company_id)
    operation = parsed["operation"]
    field = parsed.get("field")

    if operation == "count":
        return await collection.count_documents(filters)

    elif operation == "sum":
        pipeline = [{"$match": filters}, {"$group": {"_id": None, "value": {"$sum": f"${field}"}}}]
        res = await collection.aggregate(pipeline).to_list(1)
        return res[0]["value"] if res else 0

    elif operation == "avg":
        pipeline = [{"$match": filters}, {"$group": {"_id": None, "value": {"$avg": f"${field}"}}}]
        res = await collection.aggregate(pipeline).to_list(1)
        return res[0]["value"] if res else 0

    elif operation == "find":
        docs = await collection.find(filters).to_list(20)
        return [convert_document(doc) for doc in docs]

    return {"error": "Unsupported operation"}

# Format Mongo documents nicely
def convert_document(doc):
    def convert_value(val):
        if isinstance(val, ObjectId):
            return str(val)
        if isinstance(val, datetime):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        return val
    return {k: convert_value(v) for k, v in doc.items()}

# Main endpoint
@app.post("/ask", response_model=QueryResponse)
async def ask_leads(req: QueryRequest):
    try:
        parsed = await interpret_question_with_field_mapping(req.question)
        result = await execute_leads_query(parsed, req.company_id)

        # Clean query copy for response (convert ObjectId to string)
        query_view = copy.deepcopy(parsed)
        if "filters" in query_view and "company" in query_view["filters"]:
            query_view["filters"]["company"] = str(req.company_id)

        return QueryResponse(
            answer=json.dumps(result, indent=2),
            query=query_view
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
