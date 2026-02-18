import logging
from typing import Dict, Any, Type
from pydantic import BaseModel, ValidationError

log = logging.getLogger("mongo")

class ToolBase:

    name: str
    Model: Type[BaseModel]

    def __init__(self, session, telemetry):
        self.session   = session
        self.telemetry = telemetry
        self.config    = session.config

        db_name = getattr(session, "current_db_name", None) or self.config.db_name
        if not hasattr(session, "mongo") or session.mongo is None:
            raise RuntimeError("Mongo client not connected; call session.connect() first")
        self.db = session.mongo[db_name]

        self._tenant_field = "company"
        self._tenant_id    = getattr(session, "current_company_id", None)
        if self._tenant_id is None:
            raise RuntimeError(
                "No tenant ID set on session (session.current_company_id). "
                "You must populate this before calling any tool."
            )

    def run(self, raw_args: Dict[str, Any]):
        args_dict = dict(raw_args or {})
        if "database" in self.Model.__fields__ and "database" not in args_dict:
            args_dict["database"] = self.db.name

        try:
            validated = self.Model(**args_dict)
        except ValidationError as e:
            raise ValueError(f"Invalid arguments: {e}")

        if hasattr(validated, "filter"):
            filt = getattr(validated, "filter") or {}
            filt[self._tenant_field] = self._tenant_id
            setattr(validated, "filter", filt)

        if hasattr(validated, "pipeline"):
            pipeline = list(getattr(validated, "pipeline") or [])
            if not (
                pipeline
                and isinstance(pipeline[0], dict)
                and "$match" in pipeline[0]
                and self._tenant_field in pipeline[0]["$match"]
            ):
                pipeline.insert(0, {"$match": {self._tenant_field: self._tenant_id}})
            setattr(validated, "pipeline", pipeline)

        if hasattr(validated, "collection") and self.config.allowed_collections:
            coll = getattr(validated, "collection")
            if coll not in self.config.allowed_collections:
                raise ValueError(f"Collection '{coll}' not in allowed list")

        impl = (
            getattr(self, "_execute")
            if "_execute" in type(self).__dict__
            else getattr(self, "execute")
        )
        return impl(validated)

    def execute(self, args: BaseModel):
        raise NotImplementedError("Tool must implement execute() or _execute()")

    def _execute(self, args: BaseModel):
        raise NotImplementedError("Tool must implement execute() or _execute()")

    @classmethod
    def openai_schema(cls) -> Dict[str, Any]:
        """Builds the JSON‐schema for OpenAI function‐calling."""
        model_schema = cls.Model.model_json_schema()
        return {
            "name":        cls.name,
            "description": cls.__doc__ or cls.name,
            "parameters": {
                "type":       "object",
                "properties": model_schema.get("properties", {}),
                "required":   model_schema.get("required", []),
            },
        }

from .find              import FindTool
from .count             import CountTool
from .aggregate         import AggregateTool
from .collection_schema import CollectionSchemaTool
from .list_collections  import ListCollectionsTool
from .search            import SearchTool
ALL_TOOLS = [
    
    FindTool,
    CountTool,
    AggregateTool,
    CollectionSchemaTool,
    ListCollectionsTool,
    SearchTool
]
