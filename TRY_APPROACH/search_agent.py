import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv
from nl2query import MongoQuery  # Ensure your module is installed or import path is correct

# Load .env values
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Check env setup
if not MONGO_URI or not DB_NAME or not COLLECTION_NAME:
    raise EnvironmentError("Check your .env file. MONGO_URI, DB_NAME, and COLLECTION_NAME must be set.")

# Setup MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Example MongoDB schema (replace with actual)
SCHEMA = '''{
  "collections": [
    {
      "name": "leads",
      "indexes": [
        { "key": { "_id": 1 } },
        { "key": { "leadStatus": 1 } },
        { "key": { "chart": 1 } },
        { "key": { "latdec": 1, "londec": 1 } }
      ],
      "document": {
        "properties": {
          "_id": { "bsonType": "string" },
          "leadNo": { "bsonType": "string" },
          "name": { "bsonType": "string" },
          "leadStatus": { "bsonType": "string" },
          "chart": { "bsonType": "string" },
          "latdec": { "bsonType": "double" },
          "londec": { "bsonType": "double" }
        }
      }
    }
  ],
  "version": 1
}'''

# Initialize model (Choose 'Phi2' or 'T5')
query_model = MongoQuery("Phi2")

def run_query(nl_question: str):
    try:
        mongo_query_str = query_model.generate_query(SCHEMA, nl_question)
        print(f"\nGenerated MongoDB Query:\n{mongo_query_str}\n")
        
        # Evaluate safely (you can improve this further)
        mongo_query = eval(mongo_query_str)

        # Determine if it's aggregate or find
        if isinstance(mongo_query, list):
            result = list(collection.aggregate(mongo_query))
        elif isinstance(mongo_query, dict):
            result = list(collection.find(mongo_query).limit(3))
        else:
            raise ValueError("Unsupported query type.")
        
        print("Query Results:")
        print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(f"[ERROR] Failed to generate/execute query: {e}")

if __name__ == "__main__":
    print("MongoDB NL2Query Agent (Phi2/T5 Model)\nType 'exit' to quit.")
    while True:
        question = input("Ask a question:\n> ")
        if question.lower() in ["exit", "quit"]:
            break
        run_query(question)
