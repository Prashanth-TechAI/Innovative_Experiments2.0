# FILE: app.py

import os
import re
import datetime
from typing import List, Dict, Any, Optional, Union
from bson import ObjectId
from fastapi import HTTPException
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
from openai import OpenAI

# ENV SETUP
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")

# CLIENTS
openai_client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]

app = FastAPI()

# MODELS
class QueryRequest(BaseModel):
    companyId: str
    query: str

class QueryResponse(BaseModel):
    query: str
    result: Union[int, float, List[Dict], Dict, List[str]]
    query_type: str
    success: bool
    message: Optional[str] = None

# UTILITIES
def convert_bson(obj):
    """Convert BSON objects to JSON serializable format"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_bson(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_bson(i) for i in obj]
    return obj

def get_company_filter(company_id: str) -> Dict[str, ObjectId]:
    """Get base company filter for all queries"""
    try:
        return {"company": ObjectId(company_id)}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid company ID format")

def map_references(docs: List[Dict[str, Any]], company_id: str) -> List[Dict[str, Any]]:
    """Map ObjectId references to human-readable names"""
    try:
        company_filter = get_company_filter(company_id)
        
        # Create lookup maps
        broker_map = {
            str(b["_id"]): b["name"]
            for b in db["brokers"].find(company_filter, {"name": 1})
        }
        lead_map = {
            str(l["_id"]): l["name"]
            for l in db["leads"].find(company_filter, {"name": 1})
        }
        
        # Map references
        for doc in docs:
            if "broker" in doc and isinstance(doc["broker"], str):
                doc["broker"] = broker_map.get(doc["broker"], doc["broker"])
            if "lead" in doc and isinstance(doc["lead"], str):
                doc["lead"] = lead_map.get(doc["lead"], doc["lead"])
                
        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error mapping references: {str(e)}")

# QUERY CLASSIFICATION
def classify_query_type(user_query: str) -> str:
    """Classify user query into one of the supported types"""
    prompt = f"""
You are a query classifier for a real estate CRM system.

Classify the following user question into one of these categories:
- COUNT: Questions asking for counts, totals, "how many"
- AVERAGE: Questions asking for averages, means
- SEARCH: Questions looking for specific records by name or properties
- LOOKUP: Questions requiring joins or complex aggregations
- TOP: Questions asking for top N, rankings, or sorted results
- FILTER: Questions with complex filtering requirements

User question:
\"\"\"{user_query}\"\"\"

Return just one of the category names.
"""
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        # Fallback classification based on keywords
        query_lower = user_query.lower()
        if any(word in query_lower for word in ["how many", "count", "total"]):
            return "COUNT"
        elif any(word in query_lower for word in ["average", "mean", "avg"]):
            return "AVERAGE"
        elif any(word in query_lower for word in ["top", "best", "highest", "lowest"]):
            return "TOP"
        elif any(word in query_lower for word in ["name", "find", "search", "show me"]):
            return "SEARCH"
        else:
            return "FILTER"

# QUERY HANDLERS
class QueryHandler:
    def __init__(self, db, company_id: str):
        self.db = db
        self.company_id = company_id
        self.company_filter = get_company_filter(company_id)

    def extract_number(self, text: str, pattern: str) -> Optional[int]:
        """Extract number from text using regex pattern"""
        match = re.search(pattern, text)
        return int(match.group(1)) if match else None

    def extract_year(self, text: str) -> Optional[int]:
        """Extract year from text"""
        match = re.search(r'(created|generated) in (\d{4})', text)
        return int(match.group(2)) if match else None

    def build_date_filter(self, year: int) -> Dict:
        """Build date filter for specific year"""
        return {
            "$gte": datetime(year, 1, 1),
            "$lt": datetime(year + 1, 1, 1)
        }

class CountQueryHandler(QueryHandler):
    def handle(self, query: str) -> int:
        """Handle count queries for all collections with comprehensive filtering"""
        query_lower = query.lower()
        
        # Leads collection
        if "lead" in query_lower:
            return self._handle_leads_count(query_lower)
        
        # Brokers collection
        elif "broker" in query_lower:
            return self._handle_brokers_count(query_lower)
        
        # Lead assignments collection
        elif "assignment" in query_lower:
            return self._handle_assignments_count(query_lower)
        
        # Lead rotations collection
        elif "rotation" in query_lower:
            return self._handle_rotations_count(query_lower)
        
        else:
            raise HTTPException(status_code=400, detail="Unrecognized count query")

    def _extract_date_filters(self, query: str) -> Dict:
        """Extract date filters from query"""
        date_filters = {}
        
        # Year filters
        year_match = re.search(r'(?:created|generated|in|from|during)\s+(?:year\s+)?(\d{4})', query)
        if year_match:
            year = int(year_match.group(1))
            date_filters["year"] = year
        
        # Month filters (by name or number)
        month_patterns = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
            'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        
        for month_name, month_num in month_patterns.items():
            if month_name in query:
                date_filters["month"] = month_num
                break
        
        # Numeric month
        month_match = re.search(r'month\s+(\d{1,2})', query)
        if month_match:
            date_filters["month"] = int(month_match.group(1))
        
        # Date range filters
        if "today" in query:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            date_filters["range"] = {"$gte": today, "$lt": tomorrow}
        elif "yesterday" in query:
            yesterday = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            today = yesterday + timedelta(days=1)
            date_filters["range"] = {"$gte": yesterday, "$lt": today}
        elif "this week" in query:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            week_start = today - timedelta(days=today.weekday())
            week_end = week_start + timedelta(days=7)
            date_filters["range"] = {"$gte": week_start, "$lt": week_end}
        elif "this month" in query:
            today = datetime.utcnow()
            month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if today.month == 12:
                month_end = month_start.replace(year=today.year + 1, month=1)
            else:
                month_end = month_start.replace(month=today.month + 1)
            date_filters["range"] = {"$gte": month_start, "$lt": month_end}
        elif "last month" in query:
            today = datetime.utcnow()
            if today.month == 1:
                month_start = today.replace(year=today.year - 1, month=12, day=1, hour=0, minute=0, second=0, microsecond=0)
                month_end = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                month_start = today.replace(month=today.month - 1, day=1, hour=0, minute=0, second=0, microsecond=0)
                month_end = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            date_filters["range"] = {"$gte": month_start, "$lt": month_end}
        
        # Specific date patterns (YYYY-MM-DD)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
        if date_match:
            try:
                specific_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                next_day = specific_date + timedelta(days=1)
                date_filters["range"] = {"$gte": specific_date, "$lt": next_day}
            except ValueError:
                pass
        
        return date_filters

    def _apply_date_filters(self, filter_query: Dict, date_filters: Dict, date_field: str = "createdAt"):
        """Apply date filters to query"""
        if "range" in date_filters:
            filter_query[date_field] = date_filters["range"]
        elif "year" in date_filters and "month" in date_filters:
            year, month = date_filters["year"], date_filters["month"]
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1)
            else:
                end_date = datetime(year, month + 1, 1)
            filter_query[date_field] = {"$gte": start_date, "$lt": end_date}
        elif "year" in date_filters:
            year = date_filters["year"]
            filter_query[date_field] = {
                "$gte": datetime(year, 1, 1),
                "$lt": datetime(year + 1, 1, 1)
            }

    def _handle_leads_count(self, query: str) -> int:
        """Handle leads count queries with comprehensive filtering"""
        filter_query = self.company_filter.copy()
        
        # Budget filters with enhanced patterns
        budget_patterns = [
            (r'min budget\s+(?:less than|<)\s+(\d+)', lambda x: {"minBudget": {"$lt": int(x)}}),
            (r'min budget\s+(?:more than|greater than|>)\s+(\d+)', lambda x: {"minBudget": {"$gt": int(x)}}),
            (r'min budget\s+(?:at least|>=)\s+(\d+)', lambda x: {"minBudget": {"$gte": int(x)}}),
            (r'min budget\s+(?:exactly|=|is)\s+(\d+)', lambda x: {"minBudget": int(x)}),
            (r'min budget\s+between\s+(\d+)\s+and\s+(\d+)', lambda x, y: {"minBudget": {"$gte": int(x), "$lte": int(y)}}),
            (r'max budget\s+(?:less than|<)\s+(\d+)', lambda x: {"maxBudget": {"$lt": int(x)}}),
            (r'max budget\s+(?:more than|greater than|>)\s+(\d+)', lambda x: {"maxBudget": {"$gt": int(x)}}),
            (r'max budget\s+(?:at least|>=)\s+(\d+)', lambda x: {"maxBudget": {"$gte": int(x)}}),
            (r'max budget\s+(?:exactly|=|is)\s+(\d+)', lambda x: {"maxBudget": int(x)}),
            (r'max budget\s+between\s+(\d+)\s+and\s+(\d+)', lambda x, y: {"maxBudget": {"$gte": int(x), "$lte": int(y)}}),
            (r'budget\s+between\s+(\d+)\s+and\s+(\d+)', lambda x, y: {"$and": [{"minBudget": {"$gte": int(x)}}, {"maxBudget": {"$lte": int(y)}}]}),
        ]
        
        for pattern, handler in budget_patterns:
            match = re.search(pattern, query)
            if match:
                filter_update = handler(*match.groups())
                filter_query.update(filter_update)
                break
        
        # Lead Status filters
        status_patterns = [
            (r'lead status\s+"([^"]+)"', lambda x: {"leadStatus": {"$regex": f"^{x}$", "$options": "i"}}),
            (r'lead status\s+([a-zA-Z\s]+)', lambda x: {"leadStatus": {"$regex": x.strip(), "$options": "i"}}),
            (r'temporary converted', lambda: {"leadStatus": {"$regex": "temporary converted", "$options": "i"}}),
            (r'converted(?!\s+to)', lambda: {"leadStatus": {"$regex": "converted", "$options": "i"}}),
            (r'on going|ongoing', lambda: {"leadStatus": {"$regex": "on going", "$options": "i"}}),
            (r'pending', lambda: {"leadStatus": {"$regex": "pending", "$options": "i"}}),
            (r'qualified', lambda: {"leadStatus": {"$regex": "qualified", "$options": "i"}}),
            (r'unqualified', lambda: {"leadStatus": {"$regex": "unqualified", "$options": "i"}}),
        ]
        
        for pattern, handler in status_patterns:
            if re.search(pattern, query):
                filter_update = handler() if handler.__code__.co_argcount == 0 else handler(re.search(pattern, query).group(1))
                filter_query.update(filter_update)
                break
        
        # Source Type filters
        if "source type" in query:
            source_match = re.search(r'source type\s+"([^"]+)"', query)
            if source_match:
                filter_query["sourceType"] = {"$regex": source_match.group(1), "$options": "i"}
            elif "broker" in query:
                filter_query["sourceType"] = "Broker"
            elif "website" in query:
                filter_query["sourceType"] = "Website"
            elif "referral" in query:
                filter_query["sourceType"] = "Referral"
        elif "from broker" in query or "via broker" in query:
            filter_query["sourceType"] = "Broker"
        
        # Commission percent filters
        comm_patterns = [
            (r'commission\s+(?:percent|%)\s+(?:less than|<)\s+(\d+)', lambda x: {"commissionPercent": {"$lt": int(x)}}),
            (r'commission\s+(?:percent|%)\s+(?:more than|>)\s+(\d+)', lambda x: {"commissionPercent": {"$gt": int(x)}}),
            (r'commission\s+(?:percent|%)\s+(?:exactly|=|is)\s+(\d+)', lambda x: {"commissionPercent": int(x)}),
            (r'commission\s+(?:percent|%)\s+between\s+(\d+)\s+and\s+(\d+)', lambda x, y: {"commissionPercent": {"$gte": int(x), "$lte": int(y)}}),
        ]
        
        for pattern, handler in comm_patterns:
            match = re.search(pattern, query)
            if match:
                filter_update = handler(*match.groups())
                filter_query.update(filter_update)
                break
        
        # Buying Timeline filters
        timeline_patterns = [
            (r'buying timeline\s+"([^"]+)"', lambda x: {"buyingTimeline": {"$regex": x, "$options": "i"}}),
            (r'0 to 6 months|within 6 months', lambda: {"buyingTimeline": {"$regex": "0 TO 6", "$options": "i"}}),
            (r'6 to 12 months', lambda: {"buyingTimeline": {"$regex": "6 TO 12", "$options": "i"}}),
            (r'immediate|immediately', lambda: {"buyingTimeline": {"$regex": "immediate", "$options": "i"}}),
        ]
        
        for pattern, handler in timeline_patterns:
            if re.search(pattern, query):
                filter_update = handler() if handler.__code__.co_argcount == 0 else handler(re.search(pattern, query).group(1))
                filter_query.update(filter_update)
                break
        
        # Property Type filters
        if "property type" in query:
            if "commercial" in query and "residential" in query:
                filter_query["propertyType"] = {"$in": ["Commercial", "Residential"]}
            elif "commercial" in query and "industrial" in query:
                filter_query["propertyType"] = {"$in": ["Commercial", "Industrial"]}
            elif "all types" in query:
                pass  # No filter needed
            elif "commercial" in query:
                filter_query["propertyType"] = "Commercial"
            elif "residential" in query:
                filter_query["propertyType"] = "Residential"
            elif "industrial" in query:
                filter_query["propertyType"] = "Industrial"
            elif "agricultural" in query:
                filter_query["propertyType"] = "Agricultural"
        
        # Status filters
        if "active" in query and "status" in query:
            filter_query["status"] = "Active"
        elif "inactive" in query and "status" in query:
            filter_query["status"] = {"$ne": "Active"}
        
        # Rotation count filters
        rotation_patterns = [
            (r'rotation count\s+(?:less than|<)\s+(\d+)', lambda x: {"rotationCount": {"$lt": int(x)}}),
            (r'rotation count\s+(?:more than|>)\s+(\d+)', lambda x: {"rotationCount": {"$gt": int(x)}}),
            (r'rotation count\s+(?:exactly|=|is)\s+(\d+)', lambda x: {"rotationCount": int(x)}),
            (r'rotation count\s+0|no rotations', lambda: {"rotationCount": 0}),
        ]
        
        for pattern, handler in rotation_patterns:
            match = re.search(pattern, query)
            if match:
                filter_update = handler(*match.groups()) if match.groups() else handler()
                filter_query.update(filter_update)
                break
        
        # Phone number filters
        if "phone" in query:
            phone_match = re.search(r'phone\s+"([^"]+)"', query)
            if phone_match:
                filter_query["phone"] = {"$regex": phone_match.group(1), "$options": "i"}
            elif "country code" in query:
                cc_match = re.search(r'country code\s+"?([+]\d+)"?', query)
                if cc_match:
                    filter_query["countryCode"] = cc_match.group(1)
        
        # Name filters
        if "name" in query and not "name some" in query:
            name_match = re.search(r'name\s+"([^"]+)"', query)
            if name_match:
                filter_query["name"] = {"$regex": name_match.group(1), "$options": "i"}
        
        # Lead number filters
        if "lead no" in query or "lead number" in query:
            leadno_match = re.search(r'(?:lead no|lead number)\s+"?([^"\s]+)"?', query)
            if leadno_match:
                filter_query["leadNo"] = {"$regex": leadno_match.group(1), "$options": "i"}
        
        # Embedded filter
        if "embedded" in query:
            if "true" in query:
                filter_query["embedded"] = True
            elif "false" in query:
                filter_query["embedded"] = False
        
        # Date filters
        date_filters = self._extract_date_filters(query)
        self._apply_date_filters(filter_query, date_filters, "createdAt")
        
        # Last activity filters
        if "last activity" in query:
            self._apply_date_filters(filter_query, date_filters, "lastActivity")
        
        return self.db["leads"].count_documents(filter_query)

    def _handle_brokers_count(self, query: str) -> int:
        """Handle brokers count queries with comprehensive filtering"""
        filter_query = self.company_filter.copy()
        
        # Status filters
        if "active" in query and "status" not in query.replace("real estate", ""):
            filter_query["status"] = "Active"
        elif "archived" in query:
            filter_query["status"] = "Archived"
        elif "inactive" in query:
            filter_query["status"] = {"$ne": "Active"}
        
        # Name filters
        if "name" in query and not "bank name" in query:
            name_match = re.search(r'name\s+"([^"]+)"', query)
            if name_match:
                filter_query["name"] = {"$regex": name_match.group(1), "$options": "i"}
        
        # Phone filters
        if "phone" in query:
            phone_match = re.search(r'phone\s+"([^"]+)"', query)
            if phone_match:
                filter_query["phone"] = {"$regex": phone_match.group(1), "$options": "i"}
        
        # Country code filters
        if "country code" in query:
            cc_match = re.search(r'country code\s+"?([+]\d+)"?', query)
            if cc_match:
                filter_query["countryCode"] = cc_match.group(1)
        
        # Commission percent filters
        comm_patterns = [
            (r'commission\s+(?:percent|%)\s+(?:less than|<)\s+(\d+)', lambda x: {"commissionPercent": {"$lt": int(x)}}),
            (r'commission\s+(?:percent|%)\s+(?:more than|>)\s+(\d+)', lambda x: {"commissionPercent": {"$gt": int(x)}}),
            (r'commission\s+(?:percent|%)\s+(?:exactly|=|is)\s+(\d+)', lambda x: {"commissionPercent": int(x)}),
            (r'commission\s+(?:percent|%)\s+between\s+(\d+)\s+and\s+(\d+)', lambda x, y: {"commissionPercent": {"$gte": int(x), "$lte": int(y)}}),
        ]
        
        for pattern, handler in comm_patterns:
            match = re.search(pattern, query)
            if match:
                filter_update = handler(*match.groups())
                filter_query.update(filter_update)
                break
        
        # Address filters
        if "address" in query:
            addr_match = re.search(r'address\s+"([^"]+)"', query)
            if addr_match:
                filter_query["address"] = {"$regex": addr_match.group(1), "$options": "i"}
        
        # Zip code filters
        if "zip code" in query or "zipcode" in query:
            zip_match = re.search(r'(?:zip code|zipcode)\s+"?([^"\s]+)"?', query)
            if zip_match:
                filter_query["zipCode"] = zip_match.group(1)
        
        # Aadhar number filters
        if "aadhar" in query:
            aadhar_match = re.search(r'aadhar\s+"?([^"\s]+)"?', query)
            if aadhar_match:
                filter_query["aadharNo"] = {"$regex": aadhar_match.group(1), "$options": "i"}
        
        # PAN number filters
        if "pan" in query:
            pan_match = re.search(r'pan\s+"?([^"\s]+)"?', query)
            if pan_match:
                filter_query["panNo"] = {"$regex": pan_match.group(1), "$options": "i"}
        
        # Bank details filters
        if "bank name" in query:
            bank_match = re.search(r'bank name\s+"([^"]+)"', query)
            if bank_match:
                filter_query["bankDetails.bankName"] = {"$regex": bank_match.group(1), "$options": "i"}
        
        if "account type" in query:
            if "saving" in query:
                filter_query["bankDetails.bankAccountType"] = "Saving"
            elif "current" in query:
                filter_query["bankDetails.bankAccountType"] = "Current"
        
        if "ifsc" in query:
            ifsc_match = re.search(r'ifsc\s+"?([^"\s]+)"?', query)
            if ifsc_match:
                filter_query["bankDetails.ifscCode"] = {"$regex": ifsc_match.group(1), "$options": "i"}
        
        # Real estate license filters
        if "license" in query:
            license_match = re.search(r'license\s+"?([^"\s]+)"?', query)
            if license_match:
                filter_query["realEstateLicenseDetails.licenseNo"] = {"$regex": license_match.group(1), "$options": "i"}
        
        if "license status" in query:
            if "active" in query:
                filter_query["realEstateLicenseDetails.status"] = "Active"
            elif "expired" in query:
                filter_query["realEstateLicenseDetails.status"] = "Expired"
        
        # Years in real estate
        if "years in real estate" in query:
            years_patterns = [
                (r'years in real estate\s+(?:more than|>)\s+(\d+)', lambda x: {"realEstateLicenseDetails.yearStartedInRealEstate": {"$lte": datetime.now().year - int(x)}}),
                (r'years in real estate\s+(?:less than|<)\s+(\d+)', lambda x: {"realEstateLicenseDetails.yearStartedInRealEstate": {"$gt": datetime.now().year - int(x)}}),
            ]
            
            for pattern, handler in years_patterns:
                match = re.search(pattern, query)
                if match:
                    filter_update = handler(*match.groups())
                    filter_query.update(filter_update)
                    break
        
        # Date filters
        date_filters = self._extract_date_filters(query)
        self._apply_date_filters(filter_query, date_filters, "createdAt")
        
        return self.db["brokers"].count_documents(filter_query)

    def _handle_assignments_count(self, query: str) -> int:
        """Handle assignments count queries with comprehensive filtering"""
        filter_query = self.company_filter.copy()
        
        # Status filters
        if "active" in query:
            filter_query["status"] = "Active"
        elif "inactive" in query:
            filter_query["status"] = {"$ne": "Active"}
        
        # Assignment filters
        if "assigned" in query and "unassigned" not in query:
            filter_query["assignee"] = {"$ne": None}
        elif "unassigned" in query:
            filter_query["assignee"] = None
        
        # Specific assignee filters
        if "assignee" in query:
            assignee_match = re.search(r'assignee\s+"([^"]+)"', query)
            if assignee_match:
                # Need to lookup broker by name and get ObjectId
                broker = self.db["brokers"].find_one(
                    {**self.company_filter, "name": {"$regex": assignee_match.group(1), "$options": "i"}},
                    {"_id": 1}
                )
                if broker:
                    filter_query["assignee"] = broker["_id"]
        
        # Team filters
        if "team" in query:
            team_match = re.search(r'team\s+"([^"]+)"', query)
            if team_match:
                # This would require team lookup logic
                pass
        
        # Lead filters
        if "lead" in query and "assignee" not in query:
            lead_match = re.search(r'lead\s+"([^"]+)"', query)
            if lead_match:
                lead = self.db["leads"].find_one(
                    {**self.company_filter, "name": {"$regex": lead_match.group(1), "$options": "i"}},
                    {"_id": 1}
                )
                if lead:
                    filter_query["lead"] = lead["_id"]
        
        # Date filters
        date_filters = self._extract_date_filters(query)
        self._apply_date_filters(filter_query, date_filters, "createdAt")
        
        return self.db["lead-assignments"].count_documents(filter_query)

    def _handle_rotations_count(self, query: str) -> int:
        """Handle rotations count queries with comprehensive filtering"""
        filter_query = self.company_filter.copy()
        
        # Date filters for rotation date
        date_filters = self._extract_date_filters(query)
        self._apply_date_filters(filter_query, date_filters, "date")
        
        # If no specific date mentioned, check for creation date
        if not date_filters:
            self._apply_date_filters(filter_query, self._extract_date_filters(query), "createdAt")
        
        # Team filters
        if "team" in query:
            team_match = re.search(r'team\s+"([^"]+)"', query)
            if team_match:
                # Team lookup would be needed
                pass
        
        # Assignee filters
        if "assignee" in query:
            assignee_match = re.search(r'assignee\s+"([^"]+)"', query)

class AverageQueryHandler(QueryHandler):
    def handle(self, query: str) -> Dict[str, float]:
        """Handle average queries"""
        query_lower = query.lower()
        
        if "budget" in query_lower:
            pipeline = [
                {"$match": self.company_filter},
                {"$group": {
                    "_id": None,
                    "avgMinBudget": {"$avg": "$minBudget"},
                    "avgMaxBudget": {"$avg": "$maxBudget"},
                    "count": {"$sum": 1}
                }}
            ]
            result = list(self.db["leads"].aggregate(pipeline))
            return result[0] if result else {"avgMinBudget": 0, "avgMaxBudget": 0, "count": 0}
        
        elif "commission" in query_lower:
            if "broker" in query_lower:
                pipeline = [
                    {"$match": self.company_filter},
                    {"$group": {
                        "_id": None,
                        "avgCommission": {"$avg": "$commissionPercent"},
                        "count": {"$sum": 1}
                    }}
                ]
                result = list(self.db["brokers"].aggregate(pipeline))
                return result[0] if result else {"avgCommission": 0, "count": 0}
            else:
                pipeline = [
                    {"$match": self.company_filter},
                    {"$group": {
                        "_id": None,
                        "avgCommission": {"$avg": "$commissionPercent"},
                        "count": {"$sum": 1}
                    }}
                ]
                result = list(self.db["leads"].aggregate(pipeline))
                return result[0] if result else {"avgCommission": 0, "count": 0}
        
        raise HTTPException(status_code=400, detail="Unrecognized average query")

class SearchQueryHandler(QueryHandler):
    def handle(self, query: str) -> List[Dict]:
        """Handle search queries for specific records"""
        query_lower = query.lower()
        
        # Extract search term (simple approach)
        search_term = query.strip()
        for word in ["find", "search", "show", "me", "tell", "about"]:
            search_term = search_term.replace(word, "").strip()
        
        # Search in leads
        if "lead" in query_lower or not any(col in query_lower for col in ["broker", "assignment"]):
            results = list(self.db["leads"].find({
                **self.company_filter,
                "name": {"$regex": search_term, "$options": "i"}
            }).limit(10))
            return map_references(convert_bson(results), self.company_id)
        
        # Search in brokers
        elif "broker" in query_lower:
            results = list(self.db["brokers"].find({
                **self.company_filter,
                "name": {"$regex": search_term, "$options": "i"}
            }).limit(10))
            return convert_bson(results)
        
        return []

class TopQueryHandler(QueryHandler):
    def handle(self, query: str) -> List[Dict]:
        """Handle top N queries with rankings"""
        query_lower = query.lower()
        
        # Extract limit
        limit_match = re.search(r'top\s+(\d+)', query_lower)
        limit = int(limit_match.group(1)) if limit_match else 5
        
        if "broker" in query_lower and "lead" in query_lower:
            # Top brokers by lead count
            pipeline = [
                {"$match": self.company_filter},
                {"$group": {
                    "_id": "$broker",
                    "leadCount": {"$sum": 1},
                    "totalMinBudget": {"$sum": "$minBudget"},
                    "totalMaxBudget": {"$sum": "$maxBudget"}
                }},
                {"$sort": {"leadCount": -1}},
                {"$limit": limit},
                {"$lookup": {
                    "from": "brokers",
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "brokerInfo"
                }},
                {"$unwind": "$brokerInfo"},
                {"$project": {
                    "brokerName": "$brokerInfo.name",
                    "leadCount": 1,
                    "totalMinBudget": 1,
                    "totalMaxBudget": 1
                }}
            ]
            results = list(self.db["leads"].aggregate(pipeline))
            return convert_bson(results)
        
        elif "lead" in query_lower and "budget" in query_lower:
            # Top leads by budget
            sort_field = "maxBudget" if "max" in query_lower else "minBudget"
            results = list(self.db["leads"].find(
                self.company_filter,
                {"name": 1, "minBudget": 1, "maxBudget": 1, "leadStatus": 1}
            ).sort(sort_field, -1).limit(limit))
            return convert_bson(results)
        
        raise HTTPException(status_code=400, detail="Unrecognized top query")

class LookupQueryHandler(QueryHandler):
    def handle(self, query: str) -> List[Dict]:
        """Handle complex lookup queries with joins"""
        query_lower = query.lower()
        
        if "assignment" in query_lower and "broker" in query_lower:
            # Get assignments with broker details
            pipeline = [
                {"$match": self.company_filter},
                {"$lookup": {
                    "from": "leads",
                    "localField": "lead",
                    "foreignField": "_id",
                    "as": "leadInfo"
                }},
                {"$lookup": {
                    "from": "brokers",
                    "localField": "assignee",
                    "foreignField": "_id",
                    "as": "brokerInfo"
                }},
                {"$unwind": {"path": "$leadInfo", "preserveNullAndEmptyArrays": True}},
                {"$unwind": {"path": "$brokerInfo", "preserveNullAndEmptyArrays": True}},
                {"$project": {
                    "leadName": "$leadInfo.name",
                    "brokerName": "$brokerInfo.name",
                    "status": 1,
                    "createdAt": 1
                }},
                {"$limit": 20}
            ]
            results = list(self.db["lead-assignments"].aggregate(pipeline))
            return convert_bson(results)
        
        raise HTTPException(status_code=400, detail="Unrecognized lookup query")

# MAIN QUERY ROUTER
def route_query(query: str, company_id: str) -> QueryResponse:
    """Route query to appropriate handler"""
    try:
        query_type = classify_query_type(query)
        
        if query_type == "COUNT":
            handler = CountQueryHandler(db, company_id)
            result = handler.handle(query)
        elif query_type == "AVERAGE":
            handler = AverageQueryHandler(db, company_id)
            result = handler.handle(query)
        elif query_type == "SEARCH":
            handler = SearchQueryHandler(db, company_id)
            result = handler.handle(query)
        elif query_type == "TOP":
            handler = TopQueryHandler(db, company_id)
            result = handler.handle(query)
        elif query_type == "LOOKUP":
            handler = LookupQueryHandler(db, company_id)
            result = handler.handle(query)
        else:
            # Fallback to search
            handler = SearchQueryHandler(db, company_id)
            result = handler.handle(query)
            query_type = "SEARCH"
        
        return QueryResponse(
            query=query,
            result=result,
            query_type=query_type,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

# API ENDPOINTS
@app.post("/query", response_model=QueryResponse)
def process_query(req: QueryRequest):
    """Process natural language query and return structured results"""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not req.companyId.strip():
        raise HTTPException(status_code=400, detail="Company ID is required")
    
    return route_query(req.query, req.companyId)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/collections/stats/{company_id}")
def get_collection_stats(company_id: str):
    """Get basic statistics for all collections"""
    try:
        company_filter = get_company_filter(company_id)
        
        stats = {
            "leads": db["leads"].count_documents(company_filter),
            "brokers": db["brokers"].count_documents(company_filter),
            "assignments": db["lead-assignments"].count_documents(company_filter),
            "rotations": db["lead-rotations"].count_documents(company_filter)
        }
        
        return {"company_id": company_id, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)