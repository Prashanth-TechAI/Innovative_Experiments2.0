from __future__ import annotations
import logging
import re
import time
from typing import Any, Dict, List, Optional
from bson import json_util
from pydantic import BaseModel, Field, validator
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, OperationFailure
from .tool_base import ToolBase, ToolException

log = logging.getLogger("mongo.find")

_NAME_RE = re.compile(r"^[A-Za-z0-9_\\-]{1,64}$")
DEFAULT_TIMEOUT_MS = 30_000
MAX_COLLECTIONS_SCAN = 100

class FindArgs(BaseModel):
    database: Optional[str] = Field(
        default=None,
        description="Database; omitted → tenant DB",
    )
    collection: Optional[str] = Field(
        default=None,
        description="Collection to query; if omitted, tool scans multiple collections",
    )
    filter: Dict[str, Any] = Field(
        default_factory=dict,
        description="MongoDB filter",
    )
    projection: Optional[Dict[str, Any]] = Field(
        default=None,
        description="MongoDB projection document",
    )
    sort: Optional[Dict[str, int]] = Field(
        default=None,
        description='Sort spec, e.g. {"createdAt": -1}',
    )
    skip: int = Field(
        default=0,
        ge=0,
        le=10_000,
        description="Documents to skip",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=1_000,
        description="Maximum docs per collection",
    )
    stopAfterFirst: bool = Field(
        default=True,
        description="Return after first collection with matches (performance)",
    )

    @validator("database", "collection")
    def _validate_names(cls, v):
        if v and not _NAME_RE.match(v):
            raise ValueError(f"Name '{v}' must match pattern {_NAME_RE.pattern}")
        return v

    @validator("sort")
    def _validate_sort(cls, v):
        if v is None:
            return v
        if not isinstance(v, dict):
            raise ValueError("sort must be a document")
        for k, val in v.items():
            if val not in (1, -1):
                raise ValueError(f"sort value for '{k}' must be 1 or -1")
        return v

def _unwrap_ci_regex(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_unwrap_ci_regex(x) for x in obj]

    if isinstance(obj, dict):
        if set(obj) == {"$regex", "$options"} and obj["$options"] == "i":
            m = re.fullmatch(r"^\^(.*)\$$", obj["$regex"])
            if m:
                return m.group(1)
        return {k: _unwrap_ci_regex(v) for k, v in obj.items()}

    return obj

class FindTool(ToolBase):
    name = "find"
    Model = FindArgs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timeout_ms = getattr(self.config, "query_timeout_ms", DEFAULT_TIMEOUT_MS)

    def execute(self, args: FindArgs) -> Dict[str, Any]:
        start_ts = time.monotonic()
        db_name = args.database or self.db.name
        db = self.session.mongo[db_name]

        filter_doc = _unwrap_ci_regex(args.filter)

        try:
            if args.collection:
                coll_list = [args.collection]
            else:
                coll_list = self._get_collection_whitelist(db)
                if len(coll_list) > MAX_COLLECTIONS_SCAN:
                    log.warning("find: limiting scan to %d collections", MAX_COLLECTIONS_SCAN)
                    coll_list = coll_list[:MAX_COLLECTIONS_SCAN]

            results: List[Dict[str, Any]] = []
            total_docs = 0

            for coll_name in coll_list:
                coll = db[coll_name]
                docs_json = self._query_collection(coll, filter_doc, args)
                if docs_json:
                    results.append(
                        {
                            "collection": coll_name,
                            "documents": docs_json,
                            "count": len(docs_json),
                        }
                    )
                    total_docs += len(docs_json)
                    if args.stopAfterFirst:
                        break

            duration_ms = int((time.monotonic() - start_ts) * 1000)
            log.info(
                "find: db=%s scanned=%d hits=%d time=%d ms",
                db_name, len(coll_list), total_docs, duration_ms
            )

            return {
                "results": results,
                "total_documents": total_docs,
                "collections_scanned": coll_list,
                "database": db_name,
                "duration_ms": duration_ms,
            }
        except (ConnectionFailure, OperationFailure) as e:
            raise ToolException(f"Database error: {e}") from e
        except ToolException:
            raise
        except Exception as e:
            log.exception("find: unexpected failure")
            raise ToolException("Internal error in find tool") from e

    def _query_collection(
        self, coll: Collection, filter_doc: Dict[str, Any], args: FindArgs
    ) -> List[Dict[str, Any]]:
        cursor = coll.find(filter_doc, projection=args.projection)
        if args.sort:
            cursor = cursor.sort(list(args.sort.items()))
        if args.skip:
            cursor = cursor.skip(args.skip)
        cursor = cursor.limit(args.limit).max_time_ms(self._timeout_ms)

        docs_bson = list(cursor)
        return json_util.loads(json_util.dumps(docs_bson))

    def _get_collection_whitelist(self, db) -> List[str]:

        if getattr(self.config, "allowed_collections", None):
            return list(self.config.allowed_collections)

        try:
            return db.list_collection_names()
        except Exception as e:
            log.error("find: failed to list collections: %s", e)
            raise ToolException("Cannot list collections") from e

__all__ = ["FindTool", "FindArgs"]
