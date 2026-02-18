# -*- coding: utf-8 -*-
"""
Enhanced Production-ready Streamlit Chatbot for MongoDB

Key improvements:
- Dynamic schema analysis for all collections
- Flexible company field detection
- Better error handling and validation
- Enhanced prompts with collection context
- Improved UI with collection info display
- Better logging and caching
"""
import streamlit as st

# Must be first Streamlit command
st.set_page_config(page_title="MongoDB AI Chatbot", layout="wide")

import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from openai import OpenAI
from datetime import datetime
import traceback

# --- Load configuration ---
load_dotenv()
MONGODB_URI = os.getenv('MONGODB_URI')
DB_NAME = os.getenv('DB_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL = os.getenv('MODEL', '')

# Validate env vars
if not all([MONGODB_URI, DB_NAME, OPENAI_API_KEY]):
    st.error("Ensure MONGODB_URI, DB_NAME, and OPENAI_API_KEY are set in .env")
    st.stop()

# --- Initialize clients ---
@st.cache_resource
def init_clients():
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    mongo_client = MongoClient(MONGODB_URI)
    return openai_client, mongo_client

openai_client, mongo_client = init_clients()
db = mongo_client[DB_NAME]
cache_coll = db['chat-cache']

# --- Helper Functions ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_collection_schema(collection_name, sample_size=100):
    """Analyze collection schema by sampling documents"""
    try:
        collection = db[collection_name]
        
        # Get sample documents
        sample_docs = list(collection.aggregate([
            {"$sample": {"size": sample_size}},
            {"$limit": sample_size}
        ]))
        
        if not sample_docs:
            return {"fields": [], "count": 0, "sample": None}
        
        # Analyze field structure
        fields = {}
        for doc in sample_docs:
            analyze_document_fields(doc, fields)
        
        # Get collection stats
        count = collection.count_documents({})
        
        return {
            "fields": fields,
            "count": count,
            "sample": sample_docs[0] if sample_docs else None
        }
    except Exception as e:
        st.error(f"Error analyzing collection {collection_name}: {e}")
        return {"fields": [], "count": 0, "sample": None}

def analyze_document_fields(doc, fields, prefix=""):
    """Recursively analyze document fields"""
    for key, value in doc.items():
        field_path = f"{prefix}.{key}" if prefix else key
        
        if field_path not in fields:
            fields[field_path] = {
                "types": set(),
                "sample_values": [],
                "is_company_field": False
            }
        
        # Track field types
        if isinstance(value, dict):
            fields[field_path]["types"].add("object")
            analyze_document_fields(value, fields, field_path)
        elif isinstance(value, list):
            fields[field_path]["types"].add("array")
            if value and isinstance(value[0], dict):
                analyze_document_fields(value[0], fields, f"{field_path}[]")
        else:
            fields[field_path]["types"].add(type(value).__name__)
        
        # Store sample values (limit to 3)
        if len(fields[field_path]["sample_values"]) < 3:
            if not isinstance(value, (dict, list)):
                fields[field_path]["sample_values"].append(str(value))
        
        # Check if this could be a company identifier field
        if any(term in key.lower() for term in ['company', 'org', 'organization', 'tenant', 'client']):
            fields[field_path]["is_company_field"] = True

def detect_company_field(collection_name, company_id):
    """Detect the appropriate company field for filtering"""
    schema = get_collection_schema(collection_name)
    
    # Look for company-related fields
    company_fields = [
        field for field, info in schema["fields"].items() 
        if info["is_company_field"] and "." not in field  # Top-level fields only
    ]
    
    if not company_fields:
        # Fallback to common field names
        common_fields = ['company', 'companyId', 'company_id', 'organizationId', 'tenantId']
        for field in common_fields:
            if field in schema["fields"]:
                company_fields.append(field)
    
    # Default to 'company' if no specific field found
    return company_fields[0] if company_fields else 'company'

def create_company_filter(collection_name, company_id):
    """Create appropriate company filter based on field detection"""
    company_field = detect_company_field(collection_name, company_id)
    
    # Try to convert to ObjectId if it looks like one
    try:
        if len(company_id) == 24:  # ObjectId length
            filter_value = ObjectId(company_id)
        else:
            filter_value = company_id
    except:
        filter_value = company_id
    
    return {"$match": {company_field: filter_value}}

def generate_enhanced_prompt(collection_name, user_input, company_id):
    """Generate enhanced prompt with collection context"""
    schema = get_collection_schema(collection_name)
    
    # Create field summary
    field_summary = []
    for field, info in list(schema["fields"].items())[:20]:  # Limit to top 20 fields
        types = ", ".join(info["types"])
        samples = ", ".join(info["sample_values"][:2]) if info["sample_values"] else ""
        field_summary.append(f"- {field}: {types}" + (f" (e.g., {samples})" if samples else ""))
    
    system_prompt = f"""You are a MongoDB aggregation pipeline expert. 

Collection: {collection_name}
Document Count: {schema['count']}
Available Fields:
{chr(10).join(field_summary)}

Company Field: {detect_company_field(collection_name, company_id)}

Guidelines:
1. Return ONLY a JSON object with 'mongoDBQuery' (array of pipeline stages) and 'queryExplanation' (string)
2. Do NOT include the company filter stage - it will be added automatically
3. Use appropriate field names from the schema above
4. Consider data types when creating queries
5. Provide clear explanations of what each stage does
6. Handle edge cases and potential null values
7. Use efficient aggregation patterns

User Query: {user_input}"""

    return system_prompt

# --- Streamlit Layout ---

# Sidebar
st.sidebar.title("⚙️ Settings")
company_id = st.sidebar.text_input("Company ID", key="company_id_input", help="Enter your company identifier")

if not company_id:
    st.sidebar.warning("Please provide a Company ID to continue.")
    st.stop()

# Collection selection with info
collections = sorted([col for col in db.list_collection_names() if col != 'chat-cache'])
collection_name = st.sidebar.selectbox("Collection", collections, key="collection_select")

# Display collection info
if collection_name:
    with st.sidebar.expander("📊 Collection Info"):
        schema = get_collection_schema(collection_name)
        st.write(f"**Documents:** {schema['count']:,}")
        st.write(f"**Fields:** {len(schema['fields'])}")
        
        company_field = detect_company_field(collection_name, company_id)
        st.write(f"**Company Field:** `{company_field}`")
        
        # Show top fields
        top_fields = list(schema['fields'].keys())[:10]
        if top_fields:
            st.write("**Sample Fields:**")
            for field in top_fields:
                st.write(f"• `{field}`")

# Main interface
st.title("🤖 MongoDB AI Chatbot")
st.markdown("Ask natural language questions about your MongoDB data!")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        if msg['role'] == 'assistant' and 'metadata' in msg:
            # Display structured response
            st.markdown(msg['content'])
            
            # Show additional metadata in expander
            with st.expander("📝 Query Details"):
                if 'pipeline' in msg['metadata']:
                    st.code(json.dumps(msg['metadata']['pipeline'], indent=2), language='json')
                if 'execution_time' in msg['metadata']:
                    st.write(f"Execution time: {msg['metadata']['execution_time']:.2f}s")
        else:
            st.write(msg['content'])

# Chat input
user_input = st.chat_input("Ask me anything about your MongoDB data...")

if user_input:
    # Add user message
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your query..."):
            try:
                start_time = datetime.now()
                
                # Generate enhanced prompt
                system_prompt = generate_enhanced_prompt(collection_name, user_input, company_id)
                
                # Call OpenAI
                completion = openai_client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Question: {user_input}"}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(completion.choices[0].message.content)
                pipeline = result.get('mongoDBQuery', [])
                explanation = result.get('queryExplanation', 'No explanation provided.')
                
                # Add company filter
                company_filter = create_company_filter(collection_name, company_id)
                pipeline.insert(0, company_filter)
                
                # Execute pipeline
                execution_start = datetime.now()
                docs = list(db[collection_name].aggregate(pipeline))
                execution_time = (datetime.now() - execution_start).total_seconds()
                
                # Format response
                pipeline_str = json.dumps(pipeline, default=str, indent=2)
                
                # Display results
                st.markdown("### 📋 Query Explanation")
                st.write(explanation)
                
                st.markdown("### 🔧 Generated Pipeline")
                st.code(pipeline_str, language='json')
                
                st.markdown("### 📊 Results")
                if docs:
                    st.write(f"Found {len(docs)} document(s)")
                    
                    # Display in a more readable format
                    if len(docs) <= 10:  # Show all if few results
                        for i, doc in enumerate(docs, 1):
                            with st.expander(f"Document {i}"):
                                st.json(doc)
                    else:  # Show summary for many results
                        st.write("First 5 documents:")
                        for i, doc in enumerate(docs[:5], 1):
                            with st.expander(f"Document {i}"):
                                st.json(doc)
                        st.write(f"... and {len(docs) - 5} more documents")
                else:
                    st.write("No documents found matching your query.")
                
                # Performance info
                total_time = (datetime.now() - start_time).total_seconds()
                st.caption(f"Query executed in {execution_time:.2f}s (total: {total_time:.2f}s)")
                
                # Prepare response for chat history
                response_content = f"**Query:** {explanation}\n\n**Results:** Found {len(docs)} document(s)"
                
                # Store in session state with metadata
                st.session_state.messages.append({
                    'role': 'assistant', 
                    'content': response_content,
                    'metadata': {
                        'pipeline': pipeline,
                        'execution_time': execution_time,
                        'result_count': len(docs)
                    }
                })
                
            except json.JSONDecodeError as e:
                error_msg = f"Error parsing AI response: {e}"
                st.error(error_msg)
                st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
                
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
                
                # Show debug info in expander
                with st.expander("🐛 Debug Information"):
                    st.code(traceback.format_exc())
    
    # Log to cache (non-blocking)
    try:
        cache_coll.insert_one({
            'company': company_id,
            'collection': collection_name,
            'sender': 'user',
            'receiver': 'assistant',
            'message': user_input,
            'response': st.session_state.messages[-1]['content'] if st.session_state.messages else '',
            'messageType': 'Text',
            'messageStatus': 'Seen',
            'timestamp': datetime.now(),
            'model': MODEL
        })
    except Exception:
        pass  # Logging failure should not interrupt user experience

# Footer
st.markdown("---")
st.markdown("💡 **Tips:** Ask questions like 'Show me recent orders', 'What are the top products by sales', or 'Find users created this month'")