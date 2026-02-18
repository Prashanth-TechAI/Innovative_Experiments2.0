import asyncio
import logging
from fastapi import FastAPI, HTTPException, Form, Response, Request
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import MONGO_URI, DATABASE_NAME
from app.extraction import extract_clean_dataset, watch_changes
from app.utils import select_collection
from app.analysis import analyze_collection

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="HomeLead AI - Multi-Tenant Data Chatbot")
app.state.companies_data = {}
COOKIE_NAME = "selected_company"

mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[DATABASE_NAME]

class ChatResponse(BaseModel):
    result: str
    info: str


@app.on_event("startup")
async def startup():
    await sync_all_companies()
    for cid in app.state.companies_data:
        asyncio.create_task(watch_changes(cid))
    asyncio.create_task(watch_new_companies())


async def sync_all_companies():
    cursor = db['companies'].find({}, {'_id': 1, 'name': 1})
    companies = [doc async for doc in cursor]

    async def sync_one(company):
        cid = str(company['_id'])
        name = company.get('name', 'unknown')
        app.state.companies_data[cid] = {'company_name': name}
        try:
            data = await extract_clean_dataset(cid)
            app.state.companies_data[cid] = data
        except Exception as e:
            logger.error("Sync failed for %s (%s): %s", name, cid, e)

    await asyncio.gather(*(sync_one(c) for c in companies))


async def watch_new_companies():
    pipeline = [{"$match": {"operationType": "insert"}}]
    async with db["companies"].watch(pipeline) as stream:
        async for change in stream:
            company = change.get("fullDocument")
            if not company:
                continue
            cid = str(company["_id"])
            name = company.get("name", "unknown")
            app.state.companies_data[cid] = {"company_name": name}
            try:
                await extract_clean_dataset(cid)
                asyncio.create_task(watch_changes(cid))
                logger.info("New company synced: %s (%s)", name, cid)
            except Exception as e:
                logger.error("Failed to sync new company %s (%s): %s", name, cid, e)


@app.post("/select-company", response_model=ChatResponse)
async def select_company(response: Response, query: str = Form(...)):
    cid = query.strip()
    if cid not in app.state.companies_data:
        raise HTTPException(status_code=404, detail="Company not found")
    asyncio.create_task(watch_changes(cid))
    response.set_cookie(COOKIE_NAME, cid, httponly=True, max_age=86400, samesite="lax")
    return ChatResponse(
        result=f"Selected: {app.state.companies_data[cid]['company_name']}",
        info="Realtime sync activated"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, query: str = Form(...)):
    cid = request.cookies.get(COOKIE_NAME)
    if not cid or cid not in app.state.companies_data:
        raise HTTPException(status_code=400, detail="No company selected")

    question = query.strip()
    collection = select_collection(question)
    result_text = await analyze_collection(app.state.companies_data[cid]['company_name'], collection, question)

    try:
        count = int(float(result_text))
        result_text = f"There are {count} total {collection}."
    except Exception:
        pass

    return ChatResponse(result=result_text, info=f"Using {collection} for company {cid}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
