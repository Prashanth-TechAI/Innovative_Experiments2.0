import os
import json
import re
from bson import ObjectId
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta

# Load env variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not MONGO_URI or not DB_NAME or not GROQ_API_KEY:
    raise ValueError("Missing environment variables")

# DB Setup
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
leads_collection = db["leads"]

# FastAPI app
app = FastAPI(title="Lead Analytics Chatbot")

# Input model
class QueryInput(BaseModel):
    question: str
    companyId: str

# Utility: LLM response cleaner
def clean_llm_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return raw

# Utility: Make BSON serializable
def convert_bson(obj: Any) -> Any:
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_bson(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_bson(i) for i in obj]
    return obj

# Enhanced date parsing for relative dates
def parse_relative_date(date_str: str) -> datetime:
    """Parse relative dates like 'last week', 'this month', etc."""
    now = datetime.now()
    date_str = date_str.lower().strip()
    
    if 'today' in date_str:
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif 'yesterday' in date_str:
        return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif 'last week' in date_str:
        return now - timedelta(weeks=1)
    elif 'this week' in date_str:
        return now - timedelta(days=now.weekday())
    elif 'last month' in date_str:
        return now - timedelta(days=30)
    elif 'this month' in date_str:
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif 'last year' in date_str:
        return now.replace(year=now.year-1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    elif 'this year' in date_str:
        return now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        # Try to parse as ISO date
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            return now

# Enhanced query interpretation with more operations
def interpret_question_to_query(question: str) -> Dict[str, Any]:
    prompt = f"""
You are a MongoDB assistant for a lead analytics system analyzing real estate leads.

Return only valid JSON (no explanation), supporting these operations:

Basic Operations:
- "count": Count documents
- "find": Retrieve documents (limited to 50)
- "sum": Sum numeric field values
- "avg": Average of numeric field values
- "min": Minimum value of a field
- "max": Maximum value of a field

Advanced Operations:
- "group_by": Group by field and count/sum/avg (requires groupField)
- "top": Get top N values (requires field, limit optional)
- "distribution": Show value distribution of a field
- "trend": Show trends over time (requires dateField)

Response format:
{{
  "collection": "leads",
  "operation": "count|find|sum|avg|min|max|group_by|top|distribution|trend",
  "field": "<field_name>|null",
  "groupField": "<field_for_grouping>|null",
  "dateField": "createdAt|updatedAt|lastActivity|null",
  "limit": 10,
  "sortBy": "<field>|null",
  "sortOrder": 1|-1,
  "filters": {{
    "leadStatus": "Active|Converted|Temporary Converted|Lost|Follow Up|On going",
    "sourceType": "Direct|Broker|Reference|Digital Marketing",
    "buyingTimeline": "0 TO 6 months|6 TO 12 months|More than 12 months",
    "minBudget": {{ "$gte": 1000000, "$lte": 50000000 }},
    "maxBudget": {{ "$gte": 1000000, "$lte": 100000000 }},
    "commissionPercent": {{ "$gte": 1, "$lte": 10 }},
    "broker": {{ "$exists": true }},
    "createdAt": {{ "$gte": "2024-01-01T00:00:00Z", "$lte": "2024-12-31T23:59:59Z" }},
    "rotationCount": {{ "$gte": 0, "$lte": 5 }},
    "status": "Active|Inactive",
    "embedded": true|false
  }}
}}

Available fields in the leads collection:
- _id, company, leadNo, name, countryCode, phone
- secondaryCountryCode, secondaryPhone, email
- sourceType, broker, commissionPercent
- minBudget, maxBudget, buyingTimeline
- leadStatus, rotationCount, lastActivity
- status, createdAt, updatedAt, embedded

Examples:
- "How many leads do we have?" → {{"operation": "count"}}
- "What's the average budget?" → {{"operation": "avg", "field": "minBudget"}}
- "Show leads by source type" → {{"operation": "group_by", "groupField": "sourceType"}}
- "Top 5 highest budget leads" → {{"operation": "top", "field": "maxBudget", "limit": 5}}
- "Leads created this month" → {{"operation": "find", "filters": {{"createdAt": {{"$gte": "2025-01-01T00:00:00Z"}}}}}}

User question: "{question}"
"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1000
            },
            timeout=30
        )

        if response.status_code != 200:
            print(f"❌ LLM API Error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail="LLM API call failed")

        raw = response.json()["choices"][0]["message"]["content"].strip()
        raw = clean_llm_json(raw)

        return json.loads(raw)
    
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=500, detail="LLM API timeout")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
        raise HTTPException(status_code=500, detail="LLM API request failed")
    except json.JSONDecodeError as e:
        print(f"❌ LLM gave invalid JSON:\n{raw}")
        raise HTTPException(status_code=500, detail=f"Invalid LLM JSON output: {e}")
    except Exception as e:
        print(f"❌ Unexpected error in query interpretation: {e}")
        raise HTTPException(status_code=500, detail="Query interpretation failed")

# Enhanced query execution with more operations
def execute_analytical_query(company_id: str, query_plan: Dict[str, Any]) -> Any:
    try:
        collection = db[query_plan.get("collection", "leads")]
        operation = query_plan.get("operation")
        field = query_plan.get("field")
        group_field = query_plan.get("groupField")
        date_field = query_plan.get("dateField", "createdAt")
        limit = query_plan.get("limit", 10)
        sort_by = query_plan.get("sortBy")
        sort_order = query_plan.get("sortOrder", -1)
        filters = query_plan.get("filters", {}) or {}

        # Inject company filter
        try:
            filters["company"] = ObjectId(company_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid companyId format")

        # Convert string ObjectIds in filters
        for key, value in filters.items():
            if key in ["broker", "project"] and isinstance(value, str):
                try:
                    filters[key] = ObjectId(value)
                except:
                    pass  # Keep as string if not valid ObjectId

        print("🧪 Running MongoDB Query:")
        print(f"Operation: {operation}")
        print(f"Company ID: {company_id}")
        print(f"Filters: {json.dumps(filters, indent=2, default=str)}")
        
        # Debug: Check if company exists and has leads
        total_leads = collection.count_documents({"company": filters["company"]})
        print(f"🔍 Total leads for this company: {total_leads}")

        # Basic operations
        if operation == "count":
            result = collection.count_documents(filters)
            print(f"🔢 Count: {result}")
            return result

        elif operation in ("sum", "avg", "min", "max") and field:
            pipeline = [
                {"$match": filters},
                {"$group": {"_id": None, "result": {f"${operation}": f"${field}"}}}
            ]
            result = list(collection.aggregate(pipeline))
            value = result[0]["result"] if result else 0
            print(f"📊 {operation.upper()}: {value}")
            return value

        elif operation == "find":
            cursor = collection.find(filters)
            if sort_by:
                cursor = cursor.sort(sort_by, sort_order)
            cursor = cursor.limit(min(limit, 50))  # Cap at 50 for performance
            docs = [convert_bson(doc) for doc in cursor]
            print(f"📋 Found {len(docs)} documents")
            return docs

        # Advanced operations
        elif operation == "group_by" and group_field:
            pipeline = [
                {"$match": filters},
                {"$group": {
                    "_id": f"${group_field}",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}},
                {"$limit": limit}
            ]
            
            if field:  # If aggregating a specific field
                pipeline[1]["$group"]["total"] = {"$sum": f"${field}"}
                pipeline[1]["$group"]["average"] = {"$avg": f"${field}"}
            
            result = list(collection.aggregate(pipeline))
            formatted_result = [convert_bson(doc) for doc in result]
            print(f"📊 Group by {group_field}: {len(formatted_result)} groups")
            return formatted_result

        elif operation == "top" and field:
            cursor = collection.find(filters, {field: 1, "name": 1, "leadNo": 1})
            cursor = cursor.sort(field, -1).limit(limit)
            docs = [convert_bson(doc) for doc in cursor]
            print(f"🏆 Top {limit} by {field}: {len(docs)} results")
            return docs

        elif operation == "distribution" and field:
            pipeline = [
                {"$match": filters},
                {"$group": {
                    "_id": f"${field}",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}},
                {"$limit": 20}
            ]
            result = list(collection.aggregate(pipeline))
            formatted_result = [convert_bson(doc) for doc in result]
            print(f"📈 Distribution of {field}: {len(formatted_result)} values")
            return formatted_result

        elif operation == "trend" and date_field:
            pipeline = [
                {"$match": filters},
                {"$group": {
                    "_id": {
                        "year": {"$year": f"${date_field}"},
                        "month": {"$month": f"${date_field}"},
                        "day": {"$dayOfMonth": f"${date_field}"}
                    },
                    "count": {"$sum": 1}
                }},
                {"$sort": {"_id": 1}},
                {"$limit": 30}
            ]
            
            if field:  # Trend of specific field values
                pipeline[1]["$group"]["total"] = {"$sum": f"${field}"}
                pipeline[1]["$group"]["average"] = {"$avg": f"${field}"}
            
            result = list(collection.aggregate(pipeline))
            formatted_result = [convert_bson(doc) for doc in result]
            print(f"📈 Trend over time: {len(formatted_result)} data points")
            return formatted_result

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported operation '{operation}' or missing required field")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Query execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

# Enhanced response formatting
def format_response(question: str, result: Any, query_plan: Dict[str, Any]) -> str:
    """Format the result into a natural language response"""
    operation = query_plan.get("operation")
    field = query_plan.get("field")
    
    if operation == "count":
        return f"Found {result:,} leads matching your criteria."
    
    elif operation in ("sum", "avg", "min", "max"):
        formatted_value = f"₹{result:,.2f}" if "budget" in field.lower() else f"{result:,.2f}"
        return f"The {operation} {field} is {formatted_value}."
    
    elif operation == "group_by":
        if len(result) > 0:
            summary = f"Here's the breakdown by {query_plan.get('groupField')}:\n"
            for item in result[:5]:  # Show top 5
                summary += f"• {item['_id']}: {item['count']} leads\n"
            return summary.strip()
    
    elif operation == "top":
        if len(result) > 0:
            summary = f"Top {len(result)} leads by {field}:\n"
            for i, item in enumerate(result[:5], 1):
                value = item.get(field, 'N/A')
                formatted_value = f"₹{value:,.2f}" if "budget" in field.lower() else str(value)
                name = item.get('name', 'Unknown')
                summary += f"{i}. {name}: {formatted_value}\n"
            return summary.strip()
    
    elif operation == "find":
        return f"Retrieved {len(result)} leads. Data includes lead details, status, and budget information."
    
    elif operation in ("distribution", "trend"):
        return f"Analysis complete. Found {len(result)} data points for your query."
    
    return f"Analysis complete with {len(result) if isinstance(result, list) else 1} result(s)."

# Main API endpoint
@app.post("/analyze-leads")
def analyze_leads(input: QueryInput):
    try:
        print(f"\n🧠 Question: {input.question}")
        
        # Interpret the question
        query_plan = interpret_question_to_query(input.question)
        print(f"📦 Query Plan:\n{json.dumps(query_plan, indent=2)}")
        
        # Execute the query
        result = execute_analytical_query(input.companyId, query_plan)
        
        # Format natural language response
        natural_answer = format_response(input.question, result, query_plan)
        
        return JSONResponse(content=convert_bson({
            "success": True,
            "answer": natural_answer,
            "generatedQuery": query_plan,
            "data": result,
            "dataCount": len(result) if isinstance(result, list) else 1
        }))
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "An unexpected error occurred",
                "details": str(e)
            }
        )

# Debug endpoint to test company data
@app.get("/debug-company/{company_id}")
def debug_company(company_id: str):
    try:
        company_oid = ObjectId(company_id)
        
        # Get total count for this company
        total_count = leads_collection.count_documents({"company": company_oid})
        
        # Get sample document
        sample_doc = leads_collection.find_one({"company": company_oid})
        
        # Get unique values for key fields
        pipeline = [
            {"$match": {"company": company_oid}},
            {"$group": {
                "_id": None,
                "leadStatuses": {"$addToSet": "$leadStatus"},
                "sourceTypes": {"$addToSet": "$sourceType"},
                "statuses": {"$addToSet": "$status"}
            }}
        ]
        
        field_values = list(leads_collection.aggregate(pipeline))
        
        return {
            "companyId": company_id,
            "totalLeads": total_count,
            "sampleDocument": convert_bson(sample_doc) if sample_doc else None,
            "fieldValues": convert_bson(field_values[0]) if field_values else None
        }
        
    except Exception as e:
        return {"error": str(e)}

# Health check endpoint
@app.get("/health")
def health_check():
    try:
        # Test database connectivity - use client instead of db
        client.admin.command('ping')
        
        # Also test if the specific database and collection are accessible
        leads_count = leads_collection.estimated_document_count()
        
        return {
            "status": "healthy", 
            "database": "connected",
            "collection": "accessible",
            "estimated_leads": leads_count
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "database": "disconnected", 
                "error": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)