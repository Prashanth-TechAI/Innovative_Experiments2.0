import logging
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from bson import SON
from .tool_base import ToolBase

log = logging.getLogger("mongo")

class ExplainFindArgs(BaseModel):
    filter:     Dict[str, Any] = Field(default_factory=dict)
    projection: Dict[str, Any] = Field(default_factory=dict)
    sort:       Dict[str, int] = Field(default_factory=dict)
    limit:      int             = Field(default=10)

class ExplainAggregateArgs(BaseModel):
    pipeline: List[Dict[str, Any]] = Field(..., min_items=1)

class ExplainCountArgs(BaseModel):
    query: Dict[str, Any] = Field(default_factory=dict)

class MethodUnion(BaseModel):
    name:      Literal["find", "aggregate", "count"]
    arguments: Union[ExplainFindArgs, ExplainAggregateArgs, ExplainCountArgs]

class ExplainArgs(BaseModel):
    database:   Optional[str] = Field(
        default=None,
        description="Database to explain against (defaults to tenant DB)"
    )
    collection: str            = Field(..., description="Collection name")
    method:     MethodUnion    = Field(..., description="Method to explain")

class ExplainTool(ToolBase):
    name  = "explain"
    Model = ExplainArgs

    def execute(self, args: ExplainArgs) -> Dict[str, Any]:
        db_name = args.database or self.db.name
        db      = self.session.mongo[db_name]
        coll    = db[args.collection]
        log.info(
            "db=%s collection=%s op=explain method=%s",
            db_name, args.collection, args.method.name
        )
        m = args.method
        if m.name == "find":
            cursor = (
                coll
                .find(m.arguments.filter, m.arguments.projection)
                .sort(list(m.arguments.sort.items()))
                .limit(m.arguments.limit)
            )
            plan = cursor.explain()

        elif m.name == "aggregate":
            cursor = coll.aggregate(m.arguments.pipeline)
            plan   = cursor.explain()

        else:
            plan = db.command(SON([
                ("explain", {"count": args.collection, "query": m.arguments.query}),
                ("verbosity", "queryPlanner")
            ]))
        return {"result": plan}
