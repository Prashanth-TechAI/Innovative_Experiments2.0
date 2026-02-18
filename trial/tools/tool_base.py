import logging
import time
import re
from typing import Any, Dict, Type, Optional
from pydantic import BaseModel, ValidationError
from utils.company_id import make_company_filter

log = logging.getLogger("mongo")

class ToolException(Exception):
    pass

def _inject_case_insensitive(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k.startswith("$") and isinstance(v, (dict, list)):
                out[k] = v
            else:
                out[k] = _inject_case_insensitive(v)
        return out
    if isinstance(obj, list):
        return [_inject_case_insensitive(x) for x in obj]
    if isinstance(obj, str):
        pattern = re.escape(obj)
        return {"$regex": pattern, "$options": "i"}
    return obj

class ToolBase:
    name: str
    Model: Type[BaseModel]

    def __init__(self, session, telemetry):
        self.session = session
        self.telemetry = telemetry
        self.config = session.config

        db_name = getattr(session, "current_db_name", None) or self.config.db_name
        if not hasattr(session, "mongo") or session.mongo is None:
            raise RuntimeError("Mongo client not connected; call session.connect() first")
        self.db = session.mongo[db_name]

    def run(self, raw_args: Dict[str, Any]) -> Any:
        start_ts = time.monotonic()
        args_dict = dict(raw_args or {})

        if "database" in self.Model.__fields__ and "database" not in args_dict:
            args_dict["database"] = self.db.name

        try:
            validated = self.Model(**args_dict)
        except ValidationError as e:
            raise ToolException(f"Invalid arguments: {e}") from e

        tenant_id = getattr(self.session, "current_company_id", None)
        if tenant_id is None:
            raise RuntimeError("No tenant ID set on session.current_company_id.")

        if hasattr(validated, "filter") and hasattr(validated, "collection"):
            coll: Optional[str] = validated.collection
            extra = validated.filter or {}

            if coll in getattr(self.config, "non_tenant_collections", []):
                filt = _inject_case_insensitive(extra) if extra else {}
                log.debug(f"Skipping tenant-scope for global collection '{coll}'; filter={filt}")
            else:
                filt = make_company_filter(coll, tenant_id, extra)
                filt = _inject_case_insensitive(filt)
                log.debug(f"Scoped filter for tenant on collection '{coll}': {filt}")
            setattr(validated, "filter", filt)

        if hasattr(validated, "pipeline") and hasattr(validated, "collection"):
            coll: Optional[str] = validated.collection
            pipeline = list(validated.pipeline or [])

            if coll in getattr(self.config, "non_tenant_collections", []):
                log.debug(f"Skipping tenant-scope pipeline for global collection '{coll}'")
            else:
                tenant_match = make_company_filter(coll, tenant_id, None)
                first_stage = pipeline[0] if pipeline else {}
                if not (
                    isinstance(first_stage, dict)
                    and "$match" in first_stage
                    and any(key in first_stage["$match"] for key in tenant_match)
                ):
                    pipeline.insert(0, {"$match": tenant_match})
                log.debug(f"Scoped pipeline for tenant on collection '{coll}': {pipeline}")
            setattr(validated, "pipeline", pipeline)

        if hasattr(validated, "collection"):
            coll: Optional[str] = validated.collection
            allowed = getattr(self.config, "allowed_collections", None)
            if coll and allowed:
                if coll not in allowed:
                    from tools.list_collections import ListCollectionsTool
                    lc_tool = ListCollectionsTool(self.session, self.telemetry)
                    whitelist = lc_tool.run({}).get("collections", [])
                    if not whitelist:
                        raise ToolException(
                            f"No collections are currently allowed. "
                            "Check your `allowed_collections` configuration."
                        )
                    raise ToolException(
                        f"Collection '{coll}' not in allowed list. "
                        f"Allowed collections: {', '.join(whitelist)}"
                    )
        impl = (
            getattr(self, "_execute")
            if "_execute" in type(self).__dict__
            else getattr(self, "execute")
        )
        try:
            log.debug(f"Starting tool '{self.name}' with args: {validated.dict()}")
            result = impl(validated)
            duration_ms = int((time.monotonic() - start_ts) * 1000)
            log.info(f"Tool '{self.name}' succeeded in {duration_ms}ms")
            if self.telemetry:
                self.telemetry.record(
                    command=self.name,
                    duration_ms=duration_ms,
                    success=True,
                    arguments=validated.dict(),
                )
            return result

        except ToolException:
            duration_ms = int((time.monotonic() - start_ts) * 1000)
            log.warning(f"Tool '{self.name}' failed in {duration_ms}ms (user error)")
            if self.telemetry:
                self.telemetry.record(
                    command=self.name,
                    duration_ms=duration_ms,
                    success=False,
                    arguments=validated.dict(),
                )
            raise

        except Exception as e:
            duration_ms = int((time.monotonic() - start_ts) * 1000)
            log.error(f"Tool '{self.name}' errored in {duration_ms}ms: {e}", exc_info=True)
            if self.telemetry:
                self.telemetry.record(
                    command=self.name,
                    duration_ms=duration_ms,
                    success=False,
                    arguments={"error": str(e)},
                )
            raise ToolException(f"An internal error occurred in '{self.name}'") from e

    def execute(self, args: BaseModel) -> Any:
        raise NotImplementedError("Tool must implement execute() or _execute()")

    def _execute(self, args: BaseModel) -> Any:
        raise NotImplementedError("Tool must implement execute() or _execute()")

    @classmethod
    def openai_schema(cls) -> Dict[str, Any]:
        schema = cls.Model.schema()
        return {
            "name": cls.name,
            "description": (cls.__doc__ or cls.name).strip(),
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        }