# Wyze Assist - Legal Billing Data Assistant (.env version)
import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from vanna.flask import VannaFlaskApp
import tempfile
import shutil

# Load environment variables
load_dotenv()

# 1. Database Setup with temporary storage
def load_data():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        CSV_PATH = os.path.join(BASE_DIR, 'data', 'updated_synthetic_legal_billing_data copy.csv')  # Updated path
        
        # Create a temporary directory that will exist for the duration of the request
        temp_dir = tempfile.mkdtemp()
        DB_PATH = os.path.join(temp_dir, 'legal_billing.db')
        
        df = pd.read_csv(CSV_PATH)
        conn = sqlite3.connect(DB_PATH)
        df.to_sql('legal_billing', conn, if_exists='replace', index=False)
        conn.close()
        return DB_PATH
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        raise

def get_db_connection():
    try:
        db_path = load_data()
        return db_path
    except Exception as e:
        print(f"Error in get_db_connection: {str(e)}")
        raise

# 2. Vanna Initialization
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

# Initialize Vanna with error handling
try:
    vn = MyVanna(config={
        'api_key': os.getenv('OPENAI_API_KEY'),
        'model': os.getenv('MODEL_NAME', '')
    })
    
    # 3. Database Connection
    vn.connect_to_sqlite(get_db_connection())
except Exception as e:
    print(f"Error initializing Vanna: {str(e)}")
    raise

# 4. Schema Training (Cell 4 exact)
df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql IS NOT NULL")
for ddl in df_ddl['sql'].to_list():
    vn.train(ddl=ddl)

# 5. Question Training (Cell 5 exact)
training_queries = [
    ("SELECT AVG(`Work Rate`) FROM legal_billing WHERE `Matter Type` = 'Litigation'", None),
    ("SELECT `Client Name`, SUM(`Billed Amount`) FROM legal_billing WHERE `Payment Status` = 'Overdue' GROUP BY `Client Name`", None),
    ("SELECT `Timekeeper Name`, AVG(`Effective Rate`) FROM legal_billing WHERE `Timekeeper Role` = 'Partner' GROUP BY `Timekeeper Name'", None),
    (None, "This dataset tracks legal billing details, including timekeepers, matters, clients, billing rates, invoices, and financial metrics."),
    (None, "mt fullfrom is matter check when you write sql quary"),
    (None, "any user ask your name then tell your name is Wyze Assist and ai assistant of WyzeRates"),
    (None, "MT#, Matter# means Matter Number check when you write sql quary"),
    (None, "MT Name means Matter Name check when you write sql quary"),
    (None, "MT Off, Loc means Matter Office check when you write sql quary"),
    (None, "MT Dept means Matter Department check when you write sql quary"),
    (None, "MT PG, Matter PG means Matter Practice Group check when you write sql quary"),
    (None, "MT Type means Matter Type check when you write sql quary"),
    (None, "batty, billing atty means Billing Attorney - check when writing SQL queries"),
    (None, "supaty, supatty means Supervising Attorney - check when writing SQL queries"),
    (None, "respaty, respatty means Responsible Attorney - check when writing SQL queries"),
    (None, "orig atty, orig means Originating Attorney - check when writing SQL queries"),
    (None, "pro atty, proatty means Proliferating Attorney - check when writing SQL queries"),
    (None, "When users say 'Active', search for 'Open' in Matter Status"),
    (None, "batty = Billing Attorney"),
    ("SELECT * FROM legal_billing WHERE `Billing Attorney` = 'John Smith'", None)
]

for sql, doc in training_queries:
    if sql:
        vn.train(sql=sql)
    if doc:
        vn.train(documentation=doc)

# 6. Flask App Setup
app = VannaFlaskApp(
    vn,
    title="Wyze Assist",
    subtitle="Your AI-powered assistant for legal billing insights.",
    logo="data:image/svg+xml;charset=UTF-8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22160%22%20height%3D%2240%22%3E%3Ctext%20x%3D%225%22%20y%3D%2230%22%20font-family%3D%22Arial%2C%20sans-serif%22%20font-size%3D%2224%22%20font-weight%3D%22bold%22%20fill%3D%22%23ffffff%22%3EWyzeAssist%3C%2Ftext%3E%3C%2Fsvg%3E",
    allow_llm_to_see_data=True
)

port = int(os.getenv('FLASK_PORT', 8084))
print(f"\n🌐 Access the app at: http://localhost:{port}")
app.run(host='0.0.0.0', port=port)