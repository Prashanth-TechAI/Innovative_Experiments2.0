import re
import logging
from typing import Any, Dict, List, Optional
from pymongo import TEXT
from bson import ObjectId
from thefuzz import fuzz
from collections.abc import MutableMapping
from pydantic import BaseModel, Field

try:
    from .tool_base import ToolBase
except ImportError:
    ToolBase = None

logger = logging.getLogger("tools.search")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)

STATIC_COLLECTIONS = [
    "companies", "brokers", "broker-payments",
    "contractors", "contractor-payments", "general-expenses",
    "lands", "projects", "properties",
    "property-bookings", "property-payments", "rent-payments",
    "tenants", "leads", "lead-assignments", "lead-rotations",
    "lead-visited-properties", "lead-notes",
    "amenities", "cold-leads"
]

class SearchArgs(BaseModel):
    term: str = Field(..., description="Search term (e.g. 'Sonu Sharma')")
    fuzzy_threshold: Optional[int] = Field(
        80, description="Fuzzy matching threshold (0–100); higher is stricter"
    )

if ToolBase:
    class SearchTool(ToolBase):
        name = "search"
        description = (
            "Search multiple collections with full-text, regex, and fuzzy matching. "
            "Returns an array of results grouped by collection."
        )
        Model = SearchArgs

        def _ensure_text_index(self, coll_name: str) -> None:
            col = self.db[coll_name]
            idxs = col.index_information()
            has_text = any(
                any(ft == "text" for _, ft in idx.get("key", []))
                for idx in idxs.values()
            )
            if not has_text:
                logger.debug("Creating wildcard text index on %s", coll_name)
                col.create_index([("$**", TEXT)], default_language="english")

        def _flatten_with_paths(self, obj: Any, parent_key: str = "") -> List[tuple]:
            items: List[tuple] = []
            if isinstance(obj, MutableMapping):
                for k, v in obj.items():
                    nk = f"{parent_key}.{k}" if parent_key else k
                    items.extend(self._flatten_with_paths(v, nk))
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    items.extend(self._flatten_with_paths(v, f"{parent_key}[{i}]"))
            elif isinstance(obj, str) and len(obj) <= 500:
                items.append((parent_key, obj))
            return items

        def execute(self, args: SearchArgs) -> Dict[str, List[Dict[str, Any]]]:
            term       = args.term.strip()
            threshold  = args.fuzzy_threshold or 80
            company_id = ObjectId(self.session.current_company_id)

            full_regex   = re.compile(re.escape(term), re.IGNORECASE)
            tokens       = [t for t in re.split(r"\s+", term) if t]
            token_regexs = [re.compile(re.escape(t), re.IGNORECASE) for t in tokens]

            results: List[Dict[str, Any]] = []
            logger.info("SearchTool: searching for '%s' (fuzz=%d)", term, threshold)

            for coll_name in STATIC_COLLECTIONS:
                self._ensure_text_index(coll_name)
                col         = self.db[coll_name]
                base_filter = {"company": company_id}
                hits: List[Dict[str, Any]] = []
                seen_ids = set()

                for doc in col.find(
                    {**base_filter, "$text": {"$search": f"\"{term}\""}},
                    {"score": {"$meta": "textScore"}}
                ).sort([("score", {"$meta": "textScore"})]):
                    hits.append({"_id": doc["_id"], "matches":[{"path":"<full-text>","snippet":term}]})
                    seen_ids.add(doc["_id"])

                if not hits:
                    for doc in col.find(
                        {**base_filter, "$text": {"$search": term}},
                        {"score": {"$meta": "textScore"}}
                    ).sort([("score", {"$meta": "textScore"})]):
                        if doc["_id"] not in seen_ids:
                            hits.append({"_id": doc["_id"], "matches":[{"path":"<text-token>","snippet":term}]})
                            seen_ids.add(doc["_id"])

                if not hits and tokens:
                    for tok in tokens:
                        for doc in col.find(
                            {**base_filter, "$text": {"$search": tok}},
                            {"score": {"$meta": "textScore"}}
                        ).sort([("score", {"$meta": "textScore"})]):
                            if doc["_id"] not in seen_ids:
                                hits.append({"_id": doc["_id"], "matches":[{"path":"<token-text>","snippet":tok}]})
                                seen_ids.add(doc["_id"])
                        if hits:
                            break

                if not hits:
                    for doc in col.find(base_filter):
                        if doc["_id"] in seen_ids:
                            continue
                        flat = self._flatten_with_paths(doc)
                        doc_matches: List[Dict[str,str]] = []
                        for path, val in flat:
                            if full_regex.search(val):
                                doc_matches.append({"path":path,"snippet":val}); continue
                            if any(rx.search(val) for rx in token_regexs):
                                doc_matches.append({"path":path,"snippet":val}); continue
                            if fuzz.token_set_ratio(term, val) >= threshold:
                                doc_matches.append({"path":path,"snippet":val}); continue
                            if any(fuzz.token_set_ratio(tok, val) >= threshold for tok in tokens):
                                doc_matches.append({"path":path,"snippet":val}); continue
                        if doc_matches:
                            hits.append({"_id": doc["_id"], "matches":doc_matches})
                            seen_ids.add(doc["_id"])

                if hits:
                    logger.info("SearchTool: '%s' → %d hits", coll_name, len(hits))
                    results.append({"collection":coll_name,"hits":hits})
                else:
                    logger.debug("SearchTool: '%s' → 0 hits", coll_name)

            logger.info("SearchTool: total collections with hits = %d", len(results))
            return {"results": results}

if __name__ == "__main__":
    import argparse, os, json
    from dotenv import load_dotenv
    from configs.config import load_config
    from src.session import Session
    from src.telemetry import Telemetry

    load_dotenv()
    p = argparse.ArgumentParser(description="Test SearchTool standalone")
    p.add_argument("-c", "--company-id",
                   required=False,
                   default=os.getenv("COMPANY_ID"),
                   help="Company ID to scope the search")
    p.add_argument("-t", "--term",
                   required=True,
                   help="Search term (e.g. 'Sonu Sharma')")
    p.add_argument("-f", "--fuzzy-threshold",
                   type=int, default=80,
                   help="Fuzzy matching threshold (0–100)")
    p.add_argument("-o", "--output-json",
                   action="store_true",
                   help="Print raw JSON instead of formatted output")
    args = p.parse_args()

    config    = load_config()
    session   = Session(config)
    session.connect()
    session.current_company_id = args.company_id
    telemetry = Telemetry(config)

    tool = SearchTool(session, telemetry)
    raw  = tool.run({"term": args.term, "fuzzy_threshold": args.fuzzy_threshold})

    if args.output_json:
        print(json.dumps(raw, default=str, indent=2))
    else:
        for entry in raw["results"]:
            print(f"\n=== {entry['collection']} ({len(entry['hits'])} matches) ===")
            for hit in entry["hits"]:
                print(f"- ID: {hit['_id']}")
                for m in hit["matches"]:
                    print(f"    • {m['path']}: {m['snippet']}")