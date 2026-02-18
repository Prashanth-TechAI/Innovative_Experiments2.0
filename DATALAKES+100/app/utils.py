import re
import datetime
from bson import ObjectId
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

def sanitize_company_name(name: str) -> str:
    """
    Sanitize a company name for PostgreSQL schema usage.
    Converts slashes to underscores and removes unsafe characters.
    """
    s = re.sub(r'[^0-9A-Za-z]+', '_', name)
    s = re.sub(r'_+', '_', s).strip('_')
    return s.lower()


def select_collection(query: str) -> str:
    """
    Select a MongoDB collection based on natural language query keywords.
    """
    query_lower = query.lower()
    synonyms = {
        "companies": ["company", "companies", "corporation", "firm", "enterprise"],
        "projects": ["project", "projects", "initiative", "program", "assignment"],
        "properties": ["property", "properties", "asset", "real estate", "estate"],
        "lands": ["land", "lands", "acreage", "terrain", "plot"],
        "leads": ["lead", "leads", "prospect", "contact", "inquiry"],
        "rent-payments": ["rent", "rents", "payment", "payments", "lease", "remittance"],
        "tenants": ["tenant", "tenants", "occupant", "renter", "lessee"],
        "brokers": ["broker", "brokers", "agent", "intermediary", "representative"],
        "amenities": ["amenity", "amenities", "facility", "service", "feature"],
        "countries": ["country", "countries", "nation", "nations", "state", "states"]
    }
    for table, syn_list in synonyms.items():
        if any(syn in query_lower for syn in syn_list):
            return table
    return "companies"


def clean_and_convert_value(val):
    """
    Convert MongoDB types like ObjectId and datetime into clean, JSON-serializable values.
    """
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
    """
    Flatten a nested dictionary for easier SQL insertion.
    Example: {'a': {'b': 1}} → {'a_b': 1}
    """
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


# Lookup logic
def try_lookup(db, primary, fallback, value, name_field="name"):
    coll = db[primary]
    result = coll.find_one({"_id": value}, {name_field: 1})
    if result and name_field in result:
        return result[name_field]
    if fallback:
        coll = db[fallback]
        result = coll.find_one({"_id": value}, {name_field: 1})
        if result and name_field in result:
            return result[name_field]
    return None


def get_simple_name(db, collection_name, value, name_field="name"):
    try:
        if not isinstance(value, ObjectId):
            value = ObjectId(value)
    except Exception:
        return str(value)
    return try_lookup(db, collection_name, collection_name, value, name_field) or str(value)


def get_property_label(db, value):
    try:
        if not isinstance(value, ObjectId):
            value = ObjectId(value)
    except Exception:
        return str(value)
    doc = db["properties"].find_one({"_id": value})
    if not doc:
        return str(value)
    return doc.get("name") or " ".join(filter(None, [
        doc.get("propertyType"), doc.get("blockName"), doc.get("floorName")
    ])) or "UnknownProperty"


def get_amenities_names(db, value):
    def lookup(val):
        return get_simple_name(db, "amenities", val, "name")
    if isinstance(value, list):
        return ", ".join([lookup(item) for item in value])
    elif isinstance(value, str):
        tokens = [token.strip() for token in value.split(',') if token.strip()]
        return ", ".join([lookup(token) for token in tokens])
    else:
        return lookup(value)


def get_country_name(db, value):
    return get_simple_name(db, "countries", value)


def get_state_name(db, value):
    try:
        if not isinstance(value, ObjectId):
            value = ObjectId(value)
    except Exception:
        return str(value)
    result = db["countries"].find_one({"states._id": value}, {"states": 1})
    if result:
        for state in result.get("states", []):
            if str(state.get("_id")) == str(value):
                return state.get("name", str(value))
    return str(value)


def get_city_name(db, value):
    try:
        if not isinstance(value, ObjectId):
            value = ObjectId(value)
    except Exception:
        return str(value)
    result = db["countries"].find_one({"states.cities._id": value}, {"states": 1})
    if result:
        for state in result.get("states", []):
            for city in state.get("cities", []):
                if str(city.get("_id")) == str(value):
                    return city.get("name", str(value))
    return str(value)


# Lookup dispatch mapping
lookup_mapping = {
    "company":   ("simple", "companies", "name"),
    "project":   ("simple", "projects", "name"),
    "land":      ("simple", "lands", "name"),
    "property":  ("custom", get_property_label, None),
    "tenant":    ("simple", "tenants", "name"),
    "broker":    ("simple", "brokers", "name"),
    "amenities": ("custom", get_amenities_names, None),
    "country":   ("custom", get_country_name, None),
    "state":     ("custom", get_state_name, None),
    "city":      ("custom", get_city_name, None)
}


def replace_field(db, key, value):
    if key not in lookup_mapping:
        return value
    lookup_type, ref_or_func, name_field = lookup_mapping[key]
    if lookup_type == "simple":
        return get_simple_name(db, ref_or_func, value, name_field)
    elif lookup_type == "custom":
        return ref_or_func(db, value)
    return value


def replace_ids_with_names(db, doc):
    """
    Recursively replace ObjectIds in a document using the lookup mapping.
    """
    if isinstance(doc, dict):
        return {
            k: replace_field(db, k, replace_ids_with_names(db, v) if isinstance(v, (dict, list)) else v)
            for k, v in doc.items()
        }
    elif isinstance(doc, list):
        return [replace_ids_with_names(db, item) for item in doc]
    return doc
