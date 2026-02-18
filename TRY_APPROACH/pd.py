import streamlit as st
st.set_page_config(page_title="Leads Query App", layout="wide")

import os
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
from groq import Groq
import traceback
import re

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

# MongoDB connection
@st.cache_resource
def init_connection():
    return MongoClient(MONGO_URI)

client = init_connection()
db = client[DB_NAME]
leads_collection = db["leads"]

# Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Streamlit UI
st.title("🔍 Query Leads Collection (MongoDB + Groq)")
st.markdown("Ask any question about your `leads` collection using plain English (e.g., **How many leads are there?**)")

user_query = st.text_area("Enter your query:", height=100)

if st.button("Generate and Run Query"):
    if not user_query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Generating Python code using LLaMA 3..."):
            try:
                prompt = f"""
You are a Python developer. Write raw executable Python code using pymongo to query the 'leads' collection in MongoDB.

MongoDB is already connected as the variable `leads_collection`.

The schema uses camelCase field names like `leadStatus`, `minBudget`, `company`, etc.

User query:
\"{user_query}\"

Just return the code that runs the query and displays the result using st.write().
Don't use markdown, docstrings, or comments.
"""


                response = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )

                # Get LLM response
                full_response = response.choices[0].message.content.strip()

                # Extract only the code block if surrounded by ```
                match = re.search(r"```(?:python)?(.*?)```", full_response, re.DOTALL)
                code_to_run = match.group(1).strip() if match else full_response.strip()

                # Replace print() with st.write() to show output in UI
                code_to_run = code_to_run.replace("print(", "st.write(")

                # Show generated code
                st.subheader("🧠 Generated Python Code")
                st.code(code_to_run, language="python")

                # Execution context
                local_namespace = {
                    "pymongo": pymongo,
                    "MongoClient": MongoClient,
                    "client": client,
                    "db": db,
                    "leads_collection": leads_collection,
                    "st": st
                }

                # Run the code
                with st.spinner("Running the generated code..."):
                    exec(code_to_run, {}, local_namespace)

            except Exception as e:
                st.error("❌ Error while executing the generated code:")
                st.exception(e)
