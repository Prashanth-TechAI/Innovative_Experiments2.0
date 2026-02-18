import os
import json
import logging
from typing import Any, Dict, Deque, List
from collections import defaultdict, deque
from datetime import datetime, timezone
import uvicorn
import socketio
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from configs.config import load_config
from configs.logging_config import setup_logging
from src.session import Session
from src.telemetry import Telemetry
from src.server import RpcServer
from tools import ALL_TOOLS
from utils.ref_mapping import async_replace_ids_with_names
from utils.lite_llm import light_llm
from utils.app_utils import call_tool

load_dotenv()
logger = logging.getLogger("mcp.host")
logger.setLevel(logging.DEBUG)

fastapi_app = FastAPI()
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=["https://homelead.in", "http://localhost:3000"],
)
app = socketio.ASGIApp(sio, fastapi_app)

HISTORY_STORE: Dict[str, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=10))
list_collections_cache: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    company_id: str
    query: str

class ChatResponse(BaseModel):
    reply: str

def _ct(name: str, args: Dict[str, Any], company_id: str):
    return call_tool(name, args, company_id, session, rpc_server)

fastapi_app.mount("/static", StaticFiles(directory="frontend"), name="static")

@fastapi_app.get("/")
async def index():
    return FileResponse("frontend/index.html")

@fastapi_app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        route = await light_llm(req.query)
        if route == '{"route":"data"}':
            return await _run_chat(req)
        return ChatResponse(reply=route)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error in /chat")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error, please try again later"
        )

async def _run_chat(req: ChatRequest) -> ChatResponse:
    logger.info("Chat start ← %s: %s", req.company_id, req.query)
    history = list(HISTORY_STORE[req.company_id])

    def _openai_chat(msgs: List[Dict[str, Any]]):
        try:
            return fastapi_app.state.openai_client.chat.completions.create(
                model         = fastapi_app.state.openai_model,
                messages      = msgs,
                functions     = functions,
                function_call = "auto",
                timeout       = fastapi_app.state.openai_timeout,
            )
        except OpenAIError as e:
            logger.error("OpenAIError: %s", e)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM unavailable, please retry"
            )
        except Exception:
            logger.exception("Unexpected error calling OpenAI")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error processing LLM request"
            )

    global functions
    functions = [t.openai_schema() for t in rpc_server.tools.values()]
    colls = list_collections_cache["result"]
    for fn in functions:
        if fn["name"] in {"collection_schema","count","aggregate","find"}:
            fn["parameters"]["properties"]["collection"]["enum"] = colls

    today = datetime.now(timezone.utc).date().isoformat()
    date_msg = {
        "role": "system",
        "content": (
            f"Current UTC date: {today}. "
            "Use [\"{today}T00:00:00Z\",\"{today}T23:59:59Z\"] for “today”."
        )
    }
    system_msg = {
        "role": "system",
        "content": (
            "You are Homelead AI – a helpful, friendly assistant for real estate questions.\n\n"
            "**Tools Available:**\n"
            "• `list_collections()`\n"
            "• `collection_schema(collection, maxValues?)`\n"
            "• `count(collection, filter)`\n"
            "• `find(collection, filter, limit?)`\n"
            "• `aggregate(collection, pipeline)`\n"
            "• `search(term, fuzzy_threshold?)`\n\n"
            "**Guidelines:**\n"
            "1. For sales query, use the property-booking collection.\n"
        )
    }
    messages = [
        date_msg,
        system_msg,
        *history,
        {"role": "user", "content": req.query},
        {"role": "assistant", "content": None,
         "function_call": {"name": "list_collections", "arguments": "{}"}},
        {"role": "function", "name": "list_collections",
         "content": json.dumps(list_collections_cache)},
    ]

    found = False
    retries = 2

    while True:
        rsp = _openai_chat(messages).choices[0].message

        if rsp.function_call:
            name = rsp.function_call.name
            args = json.loads(rsp.function_call.arguments or "{}")

            if name == "search":
                res, empty = _ct("search", args, req.company_id)
                messages.append({"role": "function", "name": "search", "content": json.dumps(res)})
                if not empty:
                    top = res["results"][0]
                    messages.append({
                        "role": "assistant", "content": None,
                        "function_call": {
                            "name": "find",
                            "arguments": json.dumps({
                                "collection": top["collection"],
                                "filter":     {"_id": top["hits"][0]["_id"]},
                                "limit":      1
                            })
                        }
                    })
                    continue
                found = True
                continue

            coll = args.get("collection")
            if name in {"count", "find", "aggregate"} and coll:
                sca, _ = _ct("collection_schema", {"collection": coll, "maxValues": 10}, req.company_id)
                messages.extend([
                    {"role": "assistant", "content": None,
                     "function_call": {"name": "collection_schema", "arguments": json.dumps({"collection": coll, "maxValues": 10})}},
                    {"role": "function", "name": "collection_schema", "content": json.dumps(sca)},
                ])
                cnt, _ = _ct("count", {"collection": coll, "filter": {}}, req.company_id)
                messages.extend([
                    {"role": "assistant", "content": None,
                     "function_call": {"name": "count", "arguments": json.dumps({"collection": coll, "filter": {}})}},
                    {"role": "function", "name": "count", "content": json.dumps(cnt)},
                ])

            out, empty = _ct(name, args, req.company_id)
            try:
                out = await async_replace_ids_with_names(out)
            except Exception:
                logger.warning("Name replacement failed for tool output", exc_info=True)
            messages.append({"role": "function", "name": name, "content": json.dumps(out)})

            found |= not empty
            if not found and retries:
                retries -= 1
                continue
            if not found:
                messages.append({"role": "assistant", "content": "No data found—please refine your question."})
                found = True
            continue

        if not found and retries:
            retries -= 1
            messages.append({"role": "assistant", "content": "Still no data—maybe try differently?"})
            continue

        raw = rsp.content or ""
        summary = raw
        try:
            summary = _openai_chat([
                {"role": "system", "content": "Write a 4–6 line clear answer."},
                {"role": "user",   "content": f"Question: {req.query}"},
                {"role": "user",   "content": f"Data: {raw}"},
            ]).choices[0].message.content.strip()
        except Exception:
            logger.warning("Summarization failed, using raw output", exc_info=True)

        HISTORY_STORE[req.company_id].append({"role": "user",      "content": req.query})
        HISTORY_STORE[req.company_id].append({"role": "assistant", "content": summary})
        return ChatResponse(reply=summary)

@sio.event
async def connect(sid, environ):
    logger.info("Socket.IO connect: %s", sid)

@sio.event
async def disconnect(sid):
    logger.info("Socket.IO disconnect: %s", sid)

@sio.on("user_query")
async def on_user_query(sid, data):
    try:
        req = ChatRequest(**data)
        res = await _run_chat(req)
        await sio.emit("assistant_reply", {"reply": res.reply}, room=sid)

    except HTTPException as he:
        logger.info("User-visible error (%s): %s", he.status_code, he.detail)
        await sio.emit("assistant_error", {"error": he.detail}, room=sid)

    except Exception:
        logger.exception("Unexpected error in socket handler")
        await sio.emit("assistant_error", {"error": "Internal server error"}, room=sid)

@fastapi_app.on_event("startup")
def on_startup():
    global config, session, telemetry, rpc_server, list_collections_cache

    config  = load_config()
    session = Session(config); session.connect()
    telemetry = Telemetry(config)
    rpc_server = RpcServer(session, config, telemetry)
    setup_logging(rpc_server, config.log_path, config.log_level)

    for T in ALL_TOOLS:
        rpc_server.register_tool(T)
    for cap in ("logging", "streaming", "interrupt"):
        rpc_server.capability(cap)
    rpc_server.resource("config://config", lambda: config.__dict__)
    telemetry.record("server_start", 0, True)

    key = os.getenv("OPENAI_API_KEY") or config.openai_api_key
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required")
    fastapi_app.state.openai_client = OpenAI(api_key=key)
    fastapi_app.state.openai_model  = config.model_name or "gpt-4o-mini"
    fastapi_app.state.openai_timeout = getattr(config, "openai_timeout", 30)

    list_collections_cache, _ = _ct(
        "list_collections", {}, config.company_id or "000000000000000000000000"
    )
    logger.info("Host ready – model=%s", fastapi_app.state.openai_model)

@fastapi_app.on_event("shutdown")
def on_shutdown():
    telemetry.record("server_stop", 0, True)
    telemetry.shutdown()
    rpc_server.close()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
        reload=False
    )
