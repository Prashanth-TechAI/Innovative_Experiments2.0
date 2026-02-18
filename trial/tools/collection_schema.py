from pydantic import BaseModel, Field
import logging
from typing import Dict, Any, Optional
from .tool_base import ToolBase, ToolException
from src.db_schema import SCHEMAS

log = logging.getLogger("mongo.collection_schema")

class CollectionSchemaArgs(BaseModel):
    collection: str = Field(
        ..., description="One of: " + ", ".join(SCHEMAS.keys())
    )
    maxValues: int = Field(
        10, description="Max distinct values to return per field"
    )

class CollectionSchemaTool(ToolBase):
    name = "collection_schema"
    Model = CollectionSchemaArgs

    def execute(self, args: CollectionSchemaArgs) -> Dict[str, Any]:
        coll = args.collection
        if coll not in SCHEMAS:
            raise ToolException(f"Unknown collection '{coll}'")

        schema = SCHEMAS[coll]
        fields = schema.get("fields", {})
        values = schema.get("values", {})
        truncated = {
            field: values.get(field, [])[:args.maxValues] for field in fields
        }

        log.info(
            "Returning schema for '%s' (%d fields). Sampled values returned for %d fields.",
            coll, len(fields), len([v for v in truncated.values() if v])
        )
        return {
            "fields": fields,
            "values": truncated
        }
