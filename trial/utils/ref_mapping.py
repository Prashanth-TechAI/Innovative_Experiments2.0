
import os
import logging
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MDB_MCP_URI")
DB_NAME   = os.getenv("DB_NAME")  or os.getenv("MDB_MCP_DB")

if not MONGO_URI:
    logger.error("MONGO_URI is not set in environment")
    raise RuntimeError("MONGO_URI must be set in .env or environment")

if not DB_NAME:
    logger.error("DB_NAME is not set in environment")
    raise RuntimeError("DB_NAME must be set in .env or environment")

mongo_client = AsyncIOMotorClient(MONGO_URI)
db           = mongo_client[DB_NAME]

cache_simple = {}
cache_state = {}
cache_city = {}

async def try_lookup(collection_name, fallback, value: ObjectId, name_field: str = "name"):
    coll = db[collection_name]
    result = await coll.find_one({"_id": value}, {name_field: 1})
    if result and name_field in result:
        return result[name_field]
    if fallback:
        coll = db[fallback]
        result = await coll.find_one({"_id": value}, {name_field: 1})
        if result and name_field in result:
            return result[name_field]
    return None

async def get_simple_name(collection_name: str, value, name_field: str = "name"):
    try:
        if not isinstance(value, ObjectId):
            value = ObjectId(value)
    except Exception:
        logger.debug(f"[get_simple_name] Value '{value}' is not a valid ObjectId.")
        return str(value)
    key = f"{collection_name}:{str(value)}:{name_field}"
    if key in cache_simple:
        return cache_simple[key]
    fallback = None
    if collection_name in ["amenities", "countries"]:
        fallback = collection_name
    name = await try_lookup(collection_name, fallback, value, name_field)
    if name is None:
        name = str(value)
    cache_simple[key] = name
    return name

async def get_amenities_names(value):
    async def lookup_amenity(val):
        return await get_simple_name("amenities", val, "name")
    if isinstance(value, list):
        names = [await lookup_amenity(item) for item in value]
    elif isinstance(value, str):
        tokens = [token.strip() for token in value.split(',') if token.strip()]
        names = [await lookup_amenity(token) for token in tokens]
    else:
        names = [await lookup_amenity(value)]
    return ", ".join(names)

async def get_country_name(value):
    return await get_simple_name("countries", value, "name")

async def get_state_name(value):
    try:
        if not isinstance(value, ObjectId):
            value = ObjectId(value)
    except Exception:
        logger.debug(f"[get_state_name] '{value}' is not a valid ObjectId.")
        return str(value)
    key = f"state:{str(value)}"
    if key in cache_state:
        return cache_state[key]
    result = await db["countries"].find_one({"states._id": value}, {"states": 1})
    if result:
        for state in result.get("states", []):
            if str(state.get("_id")) == str(value):
                cache_state[key] = state.get("name", str(value))
                return cache_state[key]
    cache_state[key] = str(value)
    return str(value)

async def get_city_name(value):
    try:
        if not isinstance(value, ObjectId):
            value = ObjectId(value)
    except Exception:
        logger.debug(f"[get_city_name] '{value}' is not a valid ObjectId.")
        return str(value)
    key = f"city:{str(value)}"
    if key in cache_city:
        return cache_city[key]
    result = await db["countries"].find_one({"states.cities._id": value}, {"states": 1})
    if result:
        for state in result.get("states", []):
            for city in state.get("cities", []):
                if str(city.get("_id")) == str(value):
                    cache_city[key] = city.get("name", str(value))
                    return cache_city[key]
    cache_city[key] = str(value)
    return str(value)

async def get_property_label(value):
    try:
        if not isinstance(value, ObjectId):
            value = ObjectId(value)
    except Exception:
        logger.debug(f"[get_property_label] '{value}' is not a valid ObjectId.")
        return str(value)
    doc = await db["properties"].find_one({"_id": value})
    if not doc:
        return str(value)
    if "name" in doc and doc["name"]:
        return doc["name"]
    else:
        prop_type = doc.get("propertyType", "")
        block = doc.get("blockName", "")
        floor = doc.get("floorName", "")
        label_parts = [part for part in [prop_type, block, floor] if part]
        return " ".join(label_parts) if label_parts else "UnknownProperty"
    
async def get_booking_label(value):
    try:
        if not isinstance(value, ObjectId):
            value = ObjectId(value)
    except Exception:
        return str(value)
    doc = await db["property-bookings"].find_one({"_id": value})
    if not doc:
        return str(value)
    lead_id = doc.get("lead")
    booking_type = doc.get("bookingType", "")
    booking_date = doc.get("bookingDate", "")
    lead_name = await get_simple_name("leads", lead_id, "name") if lead_id else ""
    return f"{lead_name} - {booking_type} - {booking_date}" if lead_name else str(value)

ASYNC_LOOKUP_MAPPING = {
    "company":                 ("simple", "companies", "name"),
    "project":                 ("simple", "projects", "name"),
    "property":                ("custom", get_property_label, None),
    "tenant":                  ("simple", "tenants", "name"),
    "broker":                  ("simple", "brokers", "name"),
    "country":                 ("custom", get_country_name, None),
    "state":                   ("custom", get_state_name, None),
    "city":                    ("custom", get_city_name, None),
    "plan":                    ("simple", "plans", "name"),
    "category":                ("simple", "project-categories", "name"),
    "propertyUnitSubType":     ("simple", "property-unit-sub-types", "name"),
    "projectUnitSubType":      ("simple", "property-unit-sub-types", "name"),
    "bhk":                     ("simple", "bhk", "name"),
    "bhkType":                 ("simple", "bhk-types", "name"),
    "amenities":               ("custom", get_amenities_names, None),
    "bank":                    ("simple", "banks", "contactPersonDetails.fullName"),
    "bankNameId":              ("simple", "bank-names", "name"),
    "lead":                    ("simple", "leads", "name"),
    "booking":                 ("custom", get_booking_label, None),
    "user":                    ("simple", "users", "firstName"),
    "assignee":                ("simple", "users", "fullName"),
    "defaultPrimary":          ("simple", "users", "fullName"),
    "defaultSecondary":        ("simple", "users", "fullName"),
    "team":                    ("simple", "teams", "name"),
    "group":                   ("simple", "groups", "name"),
    "designation":             ("simple", "designations", "name")
}
def clean_and_convert_value(val):
    if isinstance(val, ObjectId):
        return str(val)
    elif isinstance(val, datetime.datetime):
        return val.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(val, dict):
        return {k: clean_and_convert_value(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [clean_and_convert_value(item) for item in val]
    return val

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if all(isinstance(item, dict) for item in v):
                list_str = "; ".join([
                    ", ".join(f"{nk}:{nv}" for nk, nv in sorted(flatten_dict(item).items()))
                    for item in v
                ])
                items.append((new_key, list_str))
            else:
                items.append((new_key, ", ".join(map(str, v))))
        else:
            items.append((new_key, v))
    return dict(items)

async def replace_field(key: str, value):
    if key not in ASYNC_LOOKUP_MAPPING:
        return value

    lookup_type, ref_or_func, name_field = ASYNC_LOOKUP_MAPPING[key]
    if lookup_type == "simple":
        return await get_simple_name(ref_or_func, value, name_field)
    elif lookup_type == "custom":
        return await ref_or_func(value)
    return value

async def async_replace_ids_with_names(doc):
    if isinstance(doc, dict):
        new_doc = {}
        for k, v in doc.items():
            if isinstance(v, dict):
                new_doc[k] = await async_replace_ids_with_names(v)
            elif isinstance(v, list):
                new_doc[k] = [
                    await async_replace_ids_with_names(item) if isinstance(item, dict) else await replace_field(k, item)
                    for item in v
                ]
            else:
                new_doc[k] = await replace_field(k, v)
        return new_doc
    elif isinstance(doc, list):
        return [await async_replace_ids_with_names(item) for item in doc]
    return doc