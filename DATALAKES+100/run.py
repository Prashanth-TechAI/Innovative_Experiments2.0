import os
from dotenv import load_dotenv
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_groq.chat_models import ChatGroq

# Load environment variables from .env file
load_dotenv()

# PostgreSQL connection config
POSTGRES_CONFIG = {
    'dbname': 'pandas_1',
    'user': os.environ.get('POSTGRES_USER', ''),
    'password': os.environ.get('POSTGRES_PASSWORD', ''),
    'host': os.environ.get('POSTGRES_HOST', ''),
    'port': os.environ.get('POSTGRES_PORT', ''),
}

# Construct URI and include the schema
db_uri = (
    f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}"
    f"@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['dbname']}"
)

# Create SQLDatabase object (pass schema)
db = SQLDatabase.from_uri(db_uri, schema="taskgrid_solutions", include_tables=["properties"])

llm = ChatGroq(
    model="",  # ✅ Correct model name for Groq
    api_key=os.environ.get("LLM_API_KEY")
)

# Wrap the SQL interaction in a chain
sql_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)

# Define tool to interact with the SQLDatabaseChain
tools = [
    Tool(
        name="PostgreSQL Query Tool",
        func=sql_chain.run,
        description="Use this tool to answer analytical questions from the taskgrid_solutions schema."
    )
]

# Build the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Test query
if __name__ == "__main__":
    while True:
        user_input = input("Ask an analytical question (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        try:
            response = agent.run(user_input)
            print("\n🟢 Response:\n", response)
        except Exception as e:
            print("\n🔴 Error:", e)
