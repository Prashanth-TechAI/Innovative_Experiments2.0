import json
import logging
from typing import Any, Dict, Tuple
from bson import ObjectId
from fastapi import HTTPException
from tools.tool_base import ToolException

logger = logging.getLogger("mcp.host_utils")
logger.setLevel(logging.DEBUG)
_BIG_FIELDS = {
    "images", "videos", "documents", "brochure", "qrCode",
    "govtApprovedDocuments", "layoutPlanImages",
}
_MAX_DOCS = 15

def to_json_safe(obj: Any) -> Any:
    try:
        if isinstance(obj, ObjectId):
            return str(obj)
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if isinstance(obj, (bytes, bytearray)):
            return f"<{len(obj)} bytes>"
        return obj
    except Exception as e:
        logger.exception("Error in to_json_safe for object: %r", obj)
        return repr(obj)

def trim_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    try:
        out: Dict[str, Any] = {}
        for k, v in doc.items():
            if k in ("__v",) or k in _BIG_FIELDS:
                continue
            if isinstance(v, dict):
                out[k] = trim_document(v)
            elif isinstance(v, list):
                out[k] = [
                    trim_document(x) if isinstance(x, dict) else to_json_safe(x)
                    for x in v[:10]
                ]
            else:
                out[k] = to_json_safe(v)
        return out
    except Exception as e:
        logger.exception("Error trimming document: %s", doc)
        return {}

def trim_result(tool: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if tool == "find":
            return {
                **raw,
                "results": [
                    {
                        **bucket,
                        "documents": [
                            trim_document(d) if isinstance(d, dict) else d
                            for d in bucket.get("documents", [])[:_MAX_DOCS]
                        ],
                    }
                    for bucket in raw.get("results", [])
                ],
            }

        if tool == "aggregate":
            return {
                **raw,
                "result": [
                    trim_document(d) if isinstance(d, dict) else d
                    for d in raw.get("result", [])[:_MAX_DOCS]
                ],
            }

        if tool == "search":
            return {
                "results": [
                    {
                        "collection": entry["collection"],
                        "hits": [
                            {"_id": str(h["_id"]), "matches": h["matches"]}
                            for h in entry.get("hits", [])[:_MAX_DOCS]
                        ],
                    }
                    for entry in raw.get("results", [])
                ]
            }

        return raw
    except Exception as e:
        logger.exception("Error trimming result for tool '%s': %s", tool, e)
        return raw

def result_is_empty(tool: str, result: Dict[str, Any]) -> bool:
    try:
        if tool == "count":
            return result.get("result", 0) == 0
        if tool == "find":
            return result.get("total_documents", 0) == 0
        if tool == "aggregate":
            return not result.get("result")
        if tool == "search":
            return not result.get("results")
        return False
    except Exception as e:
        logger.exception("Error checking emptiness for tool '%s': %s", tool, e)
        return False

def call_tool(
    name: str,
    args: Dict[str, Any],
    company_id: str,
    session,
    rpc_server
) -> Tuple[Dict[str, Any], bool]:
    try:
        session.current_company_id = company_id
    except Exception as e:
        logger.exception("Error setting session.current_company_id to '%s'", company_id)
        raise HTTPException(500, "Internal server error")

    if name == "find":
        filt = args.get("filter", {})
        _id = filt.get("_id")
        if isinstance(_id, str):
            try:
                filt["_id"] = ObjectId(_id)
                args["filter"] = filt
                logger.debug("Converted _id to ObjectId(%s)", _id)
            except Exception as e:
                logger.exception("Invalid _id for find: %s", _id)

    try:
        logger.info("MCP Request → %s %s", name, json.dumps(args, default=str))
    except Exception:
        logger.exception("Error logging request for tool '%s'", name)

    try:
        raw = rpc_server.call_tool(name, args)
    except ToolException as te:
        logger.error("ToolException in %s: %s", name, te)
        raise HTTPException(400, str(te))
    except Exception as e:
        logger.exception("Unexpected RPC error in %s", name)
        raise HTTPException(500, "Internal server error")

    try:
        logger.info("MCP Response ← %s %s", name, json.dumps(raw, default=str))
    except Exception:
        logger.exception("Error logging response for tool '%s'", name)

    try:
        shrunk = trim_result(name, raw)
        empty = result_is_empty(name, shrunk)
        logger.debug("Trimmed %s empty=%s", name, empty)
    except Exception as e:
        logger.exception("Error processing result for tool '%s'", name)
        shrunk, empty = raw, False
    return shrunk, empty
