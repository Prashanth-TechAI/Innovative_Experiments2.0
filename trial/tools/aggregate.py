import logging
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, root_validator
from dateutil.parser import isoparse
from bson import ObjectId

from .tool_base import ToolBase, ToolException, _inject_case_insensitive
from src.db_schema import SCHEMAS

log = logging.getLogger("mongo.aggregate")


def normalize_field_name(input_name: str, collection: str) -> str:
    """
    Convert snake_case or lowercase to camelCase based on your SCHEMAS.
    Falls back to the raw input if no match is found.
    """
    fields = SCHEMAS.get(collection, {}).get("fields", {})
    key = input_name.lower().replace("_", "")
    for schema_field in fields:
        if schema_field.lower() == key:
            return schema_field
    return input_name


def _convert_iso_dates(obj: Any) -> Any:
    """Recursively parse ISO‐8601 strings into datetime."""
    if isinstance(obj, dict):
        return {k: _convert_iso_dates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_iso_dates(v) for v in obj]
    if isinstance(obj, str):
        try:
            return isoparse(obj)
        except Exception:
            return obj
    return obj


def _sanitize_pipeline_keys(pl: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Strip whitespace from all stage keys (fixes things like ' $group').
    Recurses into sub-documents and arrays.
    """
    def clean(doc: Any) -> Any:
        if isinstance(doc, dict):
            out = {}
            for k, v in doc.items():
                clean_key = k.strip()
                out[clean_key] = clean(v)
            return out
        if isinstance(doc, list):
            return [clean(v) for v in doc]
        return doc

    return [clean(stage) for stage in pl]


class AggregateArgs(BaseModel):
    database:    Optional[str]                  = None
    collection:  str                            = Field(..., description="Name of the collection")
    pipeline:    Optional[List[Dict[str, Any]]] = Field(None, description="Full custom aggregation pipeline")
    groupBy:     Optional[Union[str, List[str]]] = Field(None, description="Field(s) to group by")
    statField:   Optional[str]                  = Field(None, description="Field for statistical op (e.g. max_budget)")
    statOp:      Optional[str]                  = Field(None, description="Aggregation op: avg, sum, min, max")
    filter:      Optional[Dict[str, Any]]       = Field(None, description="Additional match filter")
    sortBy:      Optional[str]                  = Field(None, description="Field to sort results by")
    sortDir:     Optional[str]                  = Field("desc", description="asc or desc")
    limit:       Optional[int]                  = Field(100, ge=1, description="Max documents to return")
    allowDiskUse: bool                          = Field(False, description="Allow disk usage for pipeline")

    @root_validator()
    def require_one_logic(cls, values):
        if not any([values.get("pipeline"), values.get("groupBy"), values.get("statField")]):
            raise ValueError("Must provide at least one of 'pipeline', 'groupBy' or 'statField'")
        return values


class AggregateTool(ToolBase):
    name = "aggregate"
    Model = AggregateArgs

    def execute(self, args: AggregateArgs) -> Dict[str, List[Dict[str, Any]]]:
        try:
            # --- Setup collection & tenant ID ---
            db_name = args.database or self.db.name
            coll    = self.session.mongo[db_name][args.collection]
            tenant  = getattr(self.session, "current_company_id", None)

            pipeline: List[Dict[str, Any]] = []

            # --- 1) Build initial $match (tenant + user filter) ---
            base_filter = args.filter or {}
            if tenant:
                base_filter["company"] = tenant
            match_stage = {"$match": _inject_case_insensitive(_convert_iso_dates(base_filter))}
            pipeline.append(match_stage)

            # --- 2) Custom pipeline overrides all other modes ---
            if args.pipeline:
                pipeline.extend(_sanitize_pipeline_keys(args.pipeline))

            # --- 3) groupBy + statOp mode ---
            elif args.groupBy and args.statField and args.statOp:
                gf = [args.groupBy] if isinstance(args.groupBy, str) else args.groupBy
                gf = [normalize_field_name(f, args.collection) for f in gf]
                sf = normalize_field_name(args.statField, args.collection)
                op = args.statOp.lower()
                if op not in {"avg", "sum", "min", "max"}:
                    raise ToolException(f"Unsupported statOp '{args.statOp}'")

                group_key = gf[0] if len(gf) == 1 else {f: f"${f}" for f in gf}
                pipeline.append({
                    "$group": {
                        "_id": group_key,
                        "stat": {f"${op}": f"${sf}"}
                    }
                })

                proj = {"_id": 0, "stat": 1}
                if isinstance(group_key, str):
                    proj["group"] = "$_id"
                else:
                    for f in gf:
                        proj[f] = f"$_id.{f}"
                pipeline.append({"$project": proj})

            # --- 4) statOp only (global stat) ---
            elif args.statField and args.statOp:
                sf = normalize_field_name(args.statField, args.collection)
                op = args.statOp.lower()
                if op not in {"avg", "sum", "min", "max"}:
                    raise ToolException(f"Unsupported statOp '{args.statOp}'")
                pipeline.extend([
                    {"$group":   {"_id": None, "result": {f"${op}": f"${sf}"}}},
                    {"$project": {"_id": 0, "result": 1}}
                ])

            # --- 5) legacy groupBy only (facet count) ---
            elif args.groupBy:
                gf = [args.groupBy] if isinstance(args.groupBy, str) else args.groupBy
                nf = [normalize_field_name(f, args.collection) for f in gf]
                gid = nf[0] if len(nf) == 1 else {f: f"${f}" for f in nf}

                group_stage = {"$group": {"_id": gid, "count": {"$sum": 1}}}
                project_stage = {
                    "$project": {
                        **({"field": "$_id"} if isinstance(gid, str)
                           else {f: f"$_id.{f}" for f in nf}),
                        "count": 1,
                        "_id": 0
                    }
                }
                pipeline.append({
                    "$facet": {
                        "total":   [{"$count": "total"}],
                        "byGroup": [group_stage, project_stage]
                    }
                })

            # --- 6) fallback: just a count of matching docs ---
            else:
                pipeline.append({"$count": "count"})

            # --- 7) Apply sort & limit if not a $facet pipeline ---
            is_facet = any("$facet" in stage for stage in pipeline)
            if args.sortBy and not is_facet:
                sf = normalize_field_name(args.sortBy, args.collection)
                direction = -1 if args.sortDir.lower() == "desc" else 1
                pipeline.append({"$sort": {sf: direction}})
            if args.limit and not is_facet:
                pipeline.append({"$limit": args.limit})

            # --- 8) Sanitize any stray whitespace mistakes in keys ---
            pipeline = _sanitize_pipeline_keys(pipeline)

            log.info("Running aggregation on %s.%s: %s", db_name, args.collection, pipeline)
            result = list(coll.aggregate(pipeline, allowDiskUse=args.allowDiskUse))
            return {"result": result}

        except ToolException:
            raise
        except Exception as e:
            log.error("Aggregation failed: %s", e, exc_info=True)
            raise ToolException(f"Aggregation failed: {e}")
