import os
import asyncio
import json
import logging
from typing import List, Dict
from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not set; all routing calls will use fallback.")
    _client = None
else:
    _client = OpenAI(api_key=api_key)

conversation_contexts: Dict[str, List[Dict]] = {}

def get_conversation_context(company_id: str) -> List[Dict]:
    try:
        if company_id not in conversation_contexts:
            conversation_contexts[company_id] = []
        return conversation_contexts[company_id][-3:]
    except Exception as e:
        logger.exception("Error getting conversation context for %s", company_id)
        return []

def update_conversation_context(company_id: str, query: str, response_type: str):
    try:
        if company_id not in conversation_contexts:
            conversation_contexts[company_id] = []
        conversation_contexts[company_id].append({
            "query": query,
            "type": response_type
        })
        conversation_contexts[company_id] = conversation_contexts[company_id][-10:]
    except Exception as e:
        logger.exception("Error updating conversation context for %s", company_id)

async def light_llm(query: str, company_id: str = "default") -> str:
    context = get_conversation_context(company_id)
    context_str = ""
    if context:
        context_str = "\n\nRECENT CONVERSATION CONTEXT:\n"
        for i, ctx in enumerate(context):
            context_str += f"{i+1}. User: '{ctx['query']}' (was: {ctx['type']})\n"
        context_str += "\nUse this context to understand follow-up questions.\n"

    messages = [
        {
            "role": "system",
            "content": (
                "You are HomeLead AI, a smart assistant for real estate companies.\n\n"
                "ROUTING DECISION:\n"
                "If the user wants DATA/INFORMATION from HomeLead system, respond EXACTLY:\n"
                '{"route":"data"}\n\n'
                "DATA QUERIES include:\n"
                "• Numbers/counts: 'how many leads', 'total properties', 'lead count', 'kitne', 'count'\n"
                "• Listings: 'show properties', 'list leads', 'display bookings'\n" 
                "• Status checks: 'converted leads', 'ongoing bookings', 'active tenants'\n"
                "• Searches: 'find property', 'search leads', 'get contact details'\n"
                "• Analytics: 'sales report', 'conversion rate', 'statistics'\n"
                "• Follow-ups: 'and converted?', 'what about ongoing?', 'pending ones?'\n"
                "  company based queries:\n"
                "• ANY business data request in ANY language\n\n"
                "CHAT QUERIES (respond naturally as HomeLead AI):\n"
                "• Greetings: 'hi', 'hello', 'namaste', 'ram ram', 'sat sri akal'\n"
                "• Small talk: 'how are you', 'what can you do', 'tell me about yourself'\n"
                "• Acknowledgments: 'ok', 'okay', 'fine', 'good', 'thanks'\n"
                "• General questions about HomeLead capabilities\n\n"
                "IMPORTANT RULES:\n"
                "1. Be VERY generous with data routing - when in doubt, route to data\n"
                "2. Short queries after data questions are usually follow-ups → route to data\n"
                "3. Support multiple languages (English, Hindi, Punjabi, etc.)\n"
                "4. Context matters - use conversation history to understand intent\n"
                "5. For natural chat, be helpful and friendly, mention HomeLead capabilities\n"
                f"{context_str}"
            )
        },
        {"role": "user", "content": query}
    ]
    if not _client:
        logger.warning("OpenAI client unavailable, using intelligent_fallback.")
        return intelligent_fallback(query, context)

    try:
        resp = await asyncio.to_thread(
            _client.chat.completions.create,
            model="gpt-4o-mini",
            messages=messages,
            timeout=10,
            temperature=0.1,
            max_tokens=150,
            top_p=0.9
        )
    except OpenAIError as e:
        logger.error("OpenAIError during routing: %s", e, exc_info=True)
        return intelligent_fallback(query, context)
    except Exception as e:
        logger.exception("Unexpected error during LLM routing", exc_info=True)
        return intelligent_fallback(query, context)

    try:
        router_reply = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("Failed to parse LLM response", exc_info=True)
        return intelligent_fallback(query, context)

    try:
        if router_reply == '{"route":"data"}' or '"route":"data"' in router_reply:
            update_conversation_context(company_id, query, "data")
            return '{"route":"data"}'

        update_conversation_context(company_id, query, "chat")
        return router_reply

    except Exception as e:
        logger.exception("Error updating context or returning router_reply", exc_info=True)
        return router_reply

def intelligent_fallback(query: str, context: List[Dict]) -> str:
    try:
        query_lower = query.lower().strip()
        last_was_data = bool(context and context[-1]["type"] == "data")
        strong_data_keywords = [
            'count', 'how many', 'kitne', 'total', 'number', 'ginti',
            'list', 'show', 'display', 'batao', 'dikhao',
            'converted', 'ongoing', 'active', 'pending', 'completed',
            'lead', 'property', 'tenant', 'booking', 'contact', 'sale'
        ]

        if any(keyword in query_lower for keyword in strong_data_keywords):
            return '{"route":"data"}'
        if last_was_data and len(query.split()) <= 3:
            followup_patterns = ['and', 'what about', 'how about', 'pending', 'active', 'converted']
            if any(p in query_lower for p in followup_patterns):
                return '{"route":"data"}'
        if any(greeting in query_lower for greeting in ['hi', 'hello', 'hey', 'namaste', 'ram', 'how are']):
            return "Hello! I'm HomeLead AI, ready to help with your real estate data and queries. What would you like to know?"
        return "I'm here to help! You can ask me about leads, properties, bookings, or any HomeLead data. What do you need?"
    except Exception as e:
        logger.exception("Error in intelligent_fallback for query '%s'", query)
        return "I'm here to help! You can ask me about leads, properties, bookings, or any HomeLead data. What do you need?"

async def light_llm_with_retry(query: str, company_id: str = "default", max_retries: int = 2) -> str:
    for attempt in range(max_retries + 1):
        try:
            return await light_llm(query, company_id)
        except Exception as e:
            logger.error(
                "Error in light_llm attempt %d for %s: %s",
                attempt, company_id, e, exc_info=True
            )
            if attempt == max_retries:
                context = get_conversation_context(company_id)
                return intelligent_fallback(query, context)
            await asyncio.sleep(0.5 * (attempt + 1))
    return intelligent_fallback(query, get_conversation_context(company_id))

def cleanup_old_contexts():
    try:
        pass
    except Exception as e:
        logger.exception("Error in cleanup_old_contexts")
