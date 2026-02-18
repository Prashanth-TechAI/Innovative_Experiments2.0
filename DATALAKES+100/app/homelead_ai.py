import os
import logging
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# LLM Initialization (secure via env variable in prod)
llm = ChatGroq(
    model="",
    api_key=os.environ.get("LLM_API_KEY", "")
)

# Prompt template for structured analytical responses
prompt = PromptTemplate(
    input_variables=["user_query", "company_name"],
    template="""
You are HomeLead AI, an advanced real estate analytical assistant with direct access to an enriched dataset for {company_name}.
This dataset is from PostgreSQL tables: companies, projects, properties, lands, leads, rent-payments, tenants, brokers, amenities, and countries.
All data is flattened and cleaned.

Follow these rules:
1. Identify the relevant table from keywords.
2. Answer analytics-related queries with computed results.
3. NEVER return raw data or code — only concise summaries.
4. Return numeric answers when appropriate.
5. Ask for clarification if the query is vague.

User Query: {user_query}
Your response:
"""
)

# Chain setup
homelead_chain = LLMChain(llm=llm, prompt=prompt)


def get_homelead_response(user_query: str, company_name: str) -> str:
    """Returns concise response from HomeLead AI for a query."""
    user_query = user_query.strip()
    logger.info("Query: '%s' | Company: '%s'", user_query, company_name)
    response = homelead_chain.run({
        "user_query": user_query,
        "company_name": company_name
    })
    logger.info("Response: %s", response)
    return response.strip()


# Optional test block
if __name__ == "__main__":
    test_company = "TaskGrid Solutions"
    queries = [
        "How many leads are generated in 2025?",
        "What is the total revenue from rent-payments?",
        "What is the highest max budget lead?",
        "Show me the trend for properties over time."
    ]
    for q in queries:
        print(f"\n🔍 {q}")
        print(get_homelead_response(q, test_company))
