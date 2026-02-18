from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
import os
import json
import re
from bson.json_util import dumps
from dotenv import load_dotenv
from openai import OpenAI

# --- ENV + Setup ---
load_dotenv()
client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

if not MONGO_URI or not DB_NAME:
    raise ValueError("Missing MongoDB config in environment variables")

app = FastAPI()
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Models ---
class LeadRequest(BaseModel):
    company_id: str
    query: str

class AgentResponse(BaseModel):
    status: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    query_type: Optional[str] = None
    collection: Optional[str] = None

# --- Prompt Context ---
SCHEMA_CONTEXT = """
You are a MongoDB query assistant for a real estate lead system.

Collections:
- leads
- lead-assignments
- lead-rotations
- brokers

Use these fields:
- leads: name, minBudget, maxBudget, leadStatus, sourceType, broker, createdAt, rotationCount
- lead-assignments: assignee, assignmentType, team, status, createdAt
- lead-rotations: assignee, rotationNumber, reason, createdAt
- brokers: name, phone, email, city, commissionPercent, realEstateLicenseDetails.status

Always return a MongoDB filter or aggregation pipeline as valid JSON (no markdown, no explanation).
"""

# --- Utility Functions ---
def format_obj(value):
    if isinstance(value, ObjectId): return str(value)
    if isinstance(value, datetime): return value.isoformat()
    if isinstance(value, list): return [format_obj(v) for v in value]
    if isinstance(value, dict): return {k: format_obj(v) for k, v in value.items()}
    return value

def detect_collection_advanced(query: str) -> str:
    query = query.lower()
    mapping = {
        "leads": ["lead", "client", "budget", "status", "requirement"],
        "lead-assignments": ["assign", "team", "assignee"],
        "lead-rotations": ["rotate", "transferred", "escalation"],
        "brokers": ["broker", "agent", "license", "commission", "jaipur", "city"]
    }
    scores = {col: sum(k in query for k in kws) for col, kws in mapping.items()}
    return max(scores, key=scores.get)

def detect_query_type(query: str) -> str:
    q = query.lower()
    if any(x in q for x in ["how many", "count", "number of"]): return "count"
    if any(x in q for x in ["sum", "total"]): return "sum"
    if any(x in q for x in ["average", "avg"]): return "average"
    if any(x in q for x in ["max", "highest"]): return "max"
    if any(x in q for x in ["min", "lowest"]): return "min"
    if any(x in q for x in ["group by", "breakdown"]): return "group"
    if any(x in q for x in ["top", "best", "first"]): return "top"
    return "find"

def process_date_ranges(query_str: str) -> str:
    now = datetime.now()
    ranges = {
        "today": (now.replace(hour=0), now),
        "yesterday": ((now - timedelta(days=1)).replace(hour=0), (now - timedelta(days=1)).replace(hour=23, minute=59)),
        "this week": ((now - timedelta(days=now.weekday())).replace(hour=0), now),
        "this month": (now.replace(day=1), now),
        "this year": (now.replace(month=1, day=1), now),
    }
    for key, (start, end) in ranges.items():
        if key in query_str.lower():
            query_str = re.sub(key, f'{{"$gte": "{start.isoformat()}", "$lte": "{end.isoformat()}"}}', query_str, flags=re.IGNORECASE)
    return query_str

def call_openai_advanced(question: str, collection: str, query_type: str) -> str:
    prompt = f"""
{SCHEMA_CONTEXT}

COLLECTION: {collection}
QUERY TYPE: {query_type}
QUESTION: {question}

Return only valid MongoDB filter or aggregation pipeline JSON.
"""
    try:
        response = client_ai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        result = response.choices[0].message.content.strip()
        result = result.strip("`json").strip("`")
        return result
    except Exception as e:
        logging.error(f"OpenAI call failed: {e}")
        raise HTTPException(status_code=500, detail="OpenAI API error")

def build_pipeline(collection, query_type, filter_obj, company_id):
    if isinstance(filter_obj, list):
        # already a pipeline
        for stage in filter_obj:
            if "$match" in stage:
                stage["$match"]["company"] = company_id
                break
        else:
            filter_obj.insert(0, {"$match": {"company": company_id}})
        return filter_obj

    filter_obj["company"] = company_id
    pipeline = [{"$match": filter_obj}]

    if query_type == "count":
        pipeline.append({"$count": "total"})
    elif query_type == "top":
        pipeline.append({"$sort": {"createdAt": -1}})
        pipeline.append({"$limit": 10})

    if collection == "leads":
        pipeline += [
            {"$lookup": {"from": "lead-assignments", "localField": "_id", "foreignField": "lead", "as": "assignments"}},
            {"$lookup": {"from": "lead-rotations", "localField": "_id", "foreignField": "lead", "as": "rotations"}},
            {"$addFields": {
                "totalRotations": {"$size": "$rotations"},
                "currentAssignment": {"$arrayElemAt": ["$assignments", 0]}
            }},
            {"$project": {
                "_id": 1, "name": 1, "phone": 1, "email": 1, "leadNo": 1,
                "sourceType": 1, "leadStatus": 1, "minBudget": 1, "maxBudget": 1,
                "currentAssignment.assignee": 1, "totalRotations": 1, "createdAt": 1
            }}
        ]
    elif collection == "brokers":
        pipeline.append({"$project": {
            "name": 1, "phone": 1, "email": 1, "commissionPercent": 1,
            "city": 1, "realEstateLicenseDetails.status": 1
        }})

    pipeline.append({"$sort": {"createdAt": -1}})
    return pipeline

def generate_summary(results: List[Dict], query_type: str) -> str:
    if query_type == "count":
        if results and "total" in results[0]:
            return f"Found {results[0]['total']} matching records."
        return f"Found {len(results)} matching records."
    if query_type in ["sum", "average", "min", "max"]:
        return f"{query_type.title()} value: {next((v for k, v in results[0].items() if k != '_id'), 0)}"
    if query_type == "top":
        return f"Top {len(results)} results:"
    return f"Found {len(results)} record(s)."

# --- Main Endpoint ---
@app.post("/lead-agent", response_model=AgentResponse)
async def lead_agent(request: LeadRequest):
    try:
        company_id = ObjectId(request.company_id)
        if not db.companies.find_one({"_id": company_id}):
            raise HTTPException(status_code=404, detail="Company not found")

        query = request.query.strip()
        collection_name = detect_collection_advanced(query)
        query_type = detect_query_type(query)
        collection = db[collection_name]

        raw = call_openai_advanced(query, collection_name, query_type)
        raw = process_date_ranges(raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logging.warning("Could not parse OpenAI JSON, using empty filter.")
            parsed = {}

        pipeline = build_pipeline(collection_name, query_type, parsed, company_id)
        results = list(collection.aggregate(pipeline))
        summary = generate_summary(results, query_type)

        return AgentResponse(
            status="success",
            query_type=query_type,
            collection=collection_name,
            data={
                "collection": collection_name,
                "query_type": query_type,
                "summary": summary,
                "results": format_obj(results),
                "executed_pipeline": format_obj(pipeline)
            }
        )
    except Exception as e:
        logging.exception("Unhandled error")
        return AgentResponse(status="error", error=str(e))

# --- Health Check ---
@app.get("/health")
async def health_check():
    try:
        db.companies.find_one()
        return {"status": "healthy", "time": datetime.utcnow().isoformat()}
    except:
        raise HTTPException(status_code=503, detail="Database connection failed")

@app.on_event("shutdown")
def on_shutdown():
    client.close()
    logging.info("Mongo connection closed")
    

mcp.find(domain, localedir=None, languages=None, all=False)
