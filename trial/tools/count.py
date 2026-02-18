import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from .tool_base import ToolBase

log = logging.getLogger("mongo")

class CountArgs(BaseModel):
    database:   Optional[str]       = Field(default=None, description="Database to query")
    collection: str                 = Field(...,    description="Collection to count")
    filter:     Dict[str, Any]      = Field(default_factory=dict, description="Query filter")

class CountTool(ToolBase):
    name  = "count"
    Model = CountArgs

    def execute(self, args: CountArgs) -> Dict[str, int]:
        db_name = args.database or self.db.name
        db      = self.session.mongo[db_name]
        coll    = db[args.collection]
        log.info(f"db={db_name} collection={args.collection} op=count filter={args.filter}")
        cnt = coll.count_documents(args.filter or {})
        return {"result": cnt}
