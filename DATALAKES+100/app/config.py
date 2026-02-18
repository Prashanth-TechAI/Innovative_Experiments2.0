import os

# MongoDB Connection
MONGO_URI = os.environ.get(
    "MONGO_URI")
DATABASE_NAME = os.environ.get("DATABASE_NAME", "")

# Collections to be synced from MongoDB to PostgreSQL
COLLECTIONS_TO_EXTRACT = [
    "companies",
    "leads",
    "lead-notes",
    "lead-assignments",
    "lead-rotations",
    "projects",
    "properties",
    "lands",
    "property-bookings",
    "property-payments",
    "broker-payments",
    "brokers",
    "tenants",
    "rent-payments",
    "contacts",
    "campaigns",
    "campaign-templates",
    "documents-and-priorities",
    "miscellaneous-documents",
    "general-expenses",
    "countries",
    "states",
    "cities",
    "amenities",
    "bhk",
    "bhk-types",
    "banks",
    "bank-names",
    "designations",
    "users",
    "teams",
    "groups",
    "property-unit-sub-types",
    "plans",
    "project-categories"
]

# Keyword mapping to identify collection types from natural language queries
COLLECTION_KEYWORDS ={
            "leads": ["lead", "leads", "prospect", "inquiry", "client", "customer"],
            "lead-assignments": ["assignment", "assignments", "assignee", "distribution"],
            "projects": ["project", "projects", "scheme", "site"],
            "properties": ["property", "properties", "unit", "building", "apartment"],
            "lands": ["land", "plot", "acre", "terrain"],
            "property-bookings": ["sale", "sales", "booking", "reservation", "conversion"],
            "property-payments": ["property-payment", "transaction", "installment", "revenue"],
            "broker-payments": ["broker", "commission", "agent", "broker commission"],
            "tenants": ["tenant", "renter", "leaseholder"],
            "rent-payments": ["rent", "lease", "monthly payment"],
            "brokers": ["broker", "channel partner"],
            "general-expenses": ["expense", "salary", "payout", "cost"],
        }
