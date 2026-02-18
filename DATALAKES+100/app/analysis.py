import os
import re
import asyncio
import logging
import pandas as pd
from sqlalchemy import text

from app.config import COLLECTIONS_TO_EXTRACT
from app.utils import sanitize_company_name, select_collection
from app.postgres_utils import async_session

from langchain_groq.chat_models import ChatGroq
from pandasai import SmartDataframe, SmartDatalake

logger = logging.getLogger(__name__)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notnull().mean() > 0.7:
                df[col] = converted
        except Exception as e:
            logger.warning("Skipping conversion for %s: %s", col, e)
    return df


async def load_table_dataframe(schema: str, table: str) -> pd.DataFrame:
    """
    Load an entire table from the given schema into a pandas DataFrame.
    """
    query = f'SELECT * FROM "{schema}"."{table}"'
    async with async_session() as session:
        result = await session.execute(text(query))
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=result.keys())
    return preprocess_dataframe(df)


def parse_structured_query(q: str):
    """
    Simple parser for queries like:
      'What is the email of Alice from leads?'
    """
    pattern = r"what\s+is\s+the\s+email\s+of\s+([\w\s]+)(?:\s+from\s+leads)?\s*$"
    match = re.search(pattern, q.lower())
    if match:
        lead_name = "".join(c for c in match.group(1).strip() if c.isalnum() or c.isspace())
        return "email", lead_name
    return None, None


async def run_structured_query(schema: str, table: str, field: str, name: str) -> str:
    """
    Executes a structured SQL query to fetch a single field by name.
    """
    query = f'''
        SELECT "{field}"
        FROM "{schema}"."{table}"
        WHERE "name" ILIKE :param
        LIMIT 1
    '''
    async with async_session() as session:
        result = await session.execute(text(query), {"param": f"%{name}%"})
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=result.keys())
    if not df.empty and df.iloc[0, 0]:
        return str(df.iloc[0, 0])
    return "0"


async def analyze_collection(
    company_name: str,
    collection: str,
    query: str,
    raw_output: bool = True
) -> str:
    """
    Main entrypoint: routes to structured SQL for simple lookups,
    or to pandasai/ChatGroq for free-form analytics.
    """
    # 1️⃣ Sanitize company name into a valid Postgres schema
    schema = sanitize_company_name(company_name)

    # 2️⃣ Pick the table (collection) to query
    table = collection or select_collection(query)

    # 3️⃣ Try a structured lookup (e.g. fetch an email by lead name)
    field, value = parse_structured_query(query)
    if field and value:
        try:
            return await run_structured_query(schema, table, field, value)
        except Exception as e:
            logger.error("Structured query error: %s", e)
            return "Internal error in structured query."

    # 4️⃣ Otherwise, use ChatGroq + pandasai
    llm = ChatGroq(
        model="",
        api_key=os.environ.get("LLM_API_KEY", "")
    )

    # 4a️⃣ Single-table analysis
    if table:
        df = await load_table_dataframe(schema, table)
        if df.empty:
            logger.warning("Empty table: %s.%s", schema, table)
            return f"Error: Table '{table}' is empty"
        try:
            smart_df = SmartDataframe(df, config={"llm": llm})
            return str(smart_df.chat(query)).strip()
        except Exception as e:
            logger.error("SmartDataframe error (%s): %s", table, e)
            return "Error during table analysis."

    # 4b️⃣ Multi-table fallback
    dataframes = await asyncio.gather(
        *[load_table_dataframe(schema, t) for t in COLLECTIONS_TO_EXTRACT]
    )
    smart_dfs = [
        SmartDataframe(df, config={"llm": llm})
        for df in dataframes if not df.empty
    ]
    if not smart_dfs:
        logger.warning("No non-empty tables found for %s", schema)
        return "Error: No data available for analysis."

    try:
        datalake = SmartDatalake(smart_dfs)
        return str(datalake.chat(query)).strip()
    except Exception as e:
        logger.error("SmartDatalake error: %s", e)
        return "Error during multi-table analysis."
