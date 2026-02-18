import logging
from typing import List, Optional
from pydantic import BaseModel
from .tool_base import ToolBase

log = logging.getLogger("mongo")

STATIC_COLLECTIONS: List[str] = [
    "companies",
    "plans",
    "brokers",
    "broker-payments",
    "contracts",
    "contractors",
    "contractor-payments",
    "general-expenses",
    "lands",
    "projects",
    "properties",
    "property-bookings",
    "property-payments",
    "rent-payments",
    "tenants",
    "leads",
    "lead-assignments",
    "lead-rotations",
    "lead-visited-properties",
    "lead-notes",
    "amenities",
    "cold-leads",
    "lead-visited-properties"
]
class ListCollectionsArgs(BaseModel):
    database: Optional[str] = None

class ListCollectionsTool(ToolBase):
    name  = "list_collections"
    Model = ListCollectionsArgs

    def execute(self, args: ListCollectionsArgs) -> dict:
        log.info(
            "Returning static collection list (ignoring args.database=%r)",
            args.database,
        )
        return {"result": STATIC_COLLECTIONS}
