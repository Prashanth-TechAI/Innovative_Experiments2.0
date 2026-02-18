import os
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

logger = logging.getLogger(__name__)

# PostgreSQL connection config
POSTGRES_CONFIG = {
    'dbname': os.environ.get('POSTGRES_DB', ''),
    'user': os.environ.get('POSTGRES_USER', ''),
    'password': os.environ.get('POSTGRES_PASSWORD', ''),
    'host': os.environ.get('POSTGRES_HOST', ''),
    'port': os.environ.get('POSTGRES_PORT', ''),
}

DATABASE_URL = (
    f"postgresql+asyncpg://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}"
    f"@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['dbname']}"
)

# Async SQLAlchemy engine and session
engine = create_async_engine(DATABASE_URL, pool_size=10, max_overflow=20)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_pg_pool():
    return engine


async def create_schema_if_not_exists(schema_name: str):
    async with async_session() as session:
        await session.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}";'))
        await session.commit()


async def create_table_if_not_exists(schema_name: str, table_name: str, columns: list):
    await create_schema_if_not_exists(schema_name)
    if 'original_id' not in columns:
        columns.append('original_id')

    cols_defs = ", ".join([f'"{col}" TEXT' for col in columns])
    unique_constraint = 'UNIQUE ("original_id")'
    full_table_name = f'"{schema_name}"."{table_name}"'

    create_query = f'''
        CREATE TABLE IF NOT EXISTS {full_table_name} (
            id SERIAL PRIMARY KEY,
            {cols_defs},
            {unique_constraint}
        );
    '''

    async with async_session() as session:
        try:
            await session.execute(text(create_query))
            await session.commit()
            logger.info("Table %s is ready.", full_table_name)
        except Exception as e:
            await session.rollback()
            logger.error("Error creating table %s: %s", full_table_name, e)
            raise


async def insert_documents_async(schema_name: str, table_name: str, documents: list):
    if not documents:
        return

    all_keys = set()
    for doc in documents:
        all_keys.update(doc.keys())
    all_keys = list(all_keys)

    await create_table_if_not_exists(schema_name, table_name, all_keys)

    rows = [
        {col: (str(doc.get(col)) if doc.get(col) is not None else None) for col in all_keys}
        for doc in documents
    ]

    columns_str = ", ".join([f'"{col}"' for col in all_keys])
    placeholders = ", ".join([f":{col}" for col in all_keys])
    full_table_name = f'"{schema_name}"."{table_name}"'

    update_columns = [f'"{col}" = EXCLUDED."{col}"' for col in all_keys if col != "original_id"]
    update_clause = ", ".join(update_columns)

    insert_query = f'''
        INSERT INTO {full_table_name} ({columns_str})
        VALUES ({placeholders})
        ON CONFLICT ("original_id") DO UPDATE SET {update_clause};
    '''

    async with async_session() as session:
        try:
            await session.execute(text(insert_query), rows)
            await session.commit()
            logger.info("Upserted %d records into %s", len(rows), full_table_name)
        except Exception as e:
            await session.rollback()
            logger.error("Error inserting into %s: %s", full_table_name, e)
            raise
