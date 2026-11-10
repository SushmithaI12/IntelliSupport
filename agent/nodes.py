"""Node functions for the support-agent graph, plus LLM helpers with retry handling."""
import json
import logging
import os
import re
import time
import uuid
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.knowledge_base import get_order_status, search_kb
from agent.state import AgentState

load_dotenv()
logger = logging.getLogger("supportpilot")

CATEGORIES = {"billing", "technical", "order_status", "general", "escalate"}
ANGRY_WORDS = ["angry", "furious", "terrible", "worst", "unacceptable", "ridiculous",
               "scam", "useless", "disgusting", "fed up", "awful"]
HUMAN_WORDS = ["human", "real person", "representative", "manager", "speak to an agent",
               "talk to an agent", "talk to someone"]
SENTINEL = "CANNOT_ANSWER"

CLASSIFY_PROMPT = """You are a classifier for a customer support system.
Read the customer's latest message and reply with ONLY a JSON object:
{"category": "billing|technical|order_status|general|escalate", "confidence": 0.0-1.0, "angry": true|false}
- billing: charges, refunds, payments, invoices
- technical: app, login, password, bugs
- order_status: tracking, delivery, cancel, address of an existing order
- general: shipping policy, warranty, anything else answerable from FAQs
- escalate: customer asks for a human or the issue is legal, safety or fraud related
- angry: true if the customer sounds upset or abusive"""

ANSWER_PROMPT = f"""You are a polite, concise customer support agent.
Answer using ONLY the context below. Do not invent policies, prices or dates.
If the context does not contain the answer, reply with exactly: {SENTINEL}

Context:
{{context}}"""


class LLMError(Exception):
    """Raised when the LLM fails after all retries."""


@lru_cache(maxsize=1)
def get_llm():
    from langchain_groq import ChatGroq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is missing. Add it to the .env file.")
    return ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                    api_key=key, temperature=0)


def safe_invoke(messages, retries: int = 2, delay: float = 1.0) -> str:
    """Call the LLM with retries. Raises LLMError on repeated failure or empty output."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            content = get_llm().invoke(messages).content
            text = content if isinstance(content, str) else str(content)
            if text.strip():
                return text.strip()
            last_error = ValueError("Empty response from model")
        except Exception as exc:  # network, rate limit, auth, model errors
            last_error = exc
            logger.warning("LLM attempt %d failed: %s", attempt + 1, exc)
        if attempt < retries:
            time.sleep(delay)
    raise LLMError(str(last_error))


def _last_user_text(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


def _heuristic_classify(text: str) -> dict:
    """Keyword fallback used when the LLM classifier fails."""
    low = text.lower()
    if any(w in low for w in ["order", "track", "deliver", "cancel", "ord-"]):
        cat = "order_status"
    elif any(w in low for w in ["refund", "charge", "payment", "bill", "invoice"]):
        cat = "billing"
    elif any(w in low for w in ["password", "login", "app", "crash", "error", "bug"]):
        cat = "technical"
    else:
        cat = "general"
    return {"category": cat, "confidence": 0.5, "angry": any(w in low for w in ANGRY_WORDS)}


def classify_node(state: AgentState) -> dict:
    """Step 1: classify the message. Also resets per-turn fields."""
    text = _last_user_text(state)
    reset = {"escalated": False, "ticket_id": None, "error": None, "draft": "", "context": ""}
    low = text.lower()

    if any(w in low for w in HUMAN_WORDS):  # explicit request, no LLM needed
        return {**reset, "category": "escalate", "confidence": 1.0,
                "angry": any(w in low for w in ANGRY_WORDS)}
    try:
        raw = safe_invoke([SystemMessage(content=CLASSIFY_PROMPT), *state["messages"][-4:]])  # recent history helps follow-ups
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        cat = str(data.get("category", "general")).lower()
        result = {"category": cat if cat in CATEGORIES else "general",
                  "confidence": float(data.get("confidence", 0.5)),
                  "angry": bool(data.get("angry", False))}
    except Exception as exc:
        logger.warning("Classification fallback used: %s", exc)
        result = _heuristic_classify(text)
    result["angry"] = result["angry"] or any(w in low for w in ANGRY_WORDS)
    return {**reset, **result}


def order_lookup_node(state: AgentState) -> dict:
    """Tool step: find an order ID in the conversation and look it up."""
    order_id = None
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            match = re.search(r"ORD-?(\d{4})", m.content, re.I)
            if match:
                order_id = f"ORD-{match.group(1)}"
                break
    if not order_id:
        ctx = "No order ID was given. Ask the customer for their order ID (format ORD-1001)."
    else:
        info = get_order_status(order_id)
        ctx = (f"Order {order_id}: status {info['status']}, carrier {info['carrier']}, "
               f"expected date {info['eta']}." if info
               else f"No order was found for {order_id}. Ask the customer to double-check the ID.")
    return {"context": ctx}


def retrieve_node(state: AgentState) -> dict:
    """Retrieve the most relevant FAQ entries for the latest message."""
    hits = search_kb(_last_user_text(state))
    return {"context": "\n\n".join(f"Q: {h['q']}\nA: {h['a']}" for h in hits)}


def generate_node(state: AgentState) -> dict:
    """Generate an answer from the retrieved context, with conversation memory."""
    if not state.get("context"):  # nothing relevant found: skip the LLM, quality check escalates
        return {"draft": SENTINEL}
    history = state["messages"][-8:]
    try:
        draft = safe_invoke([SystemMessage(content=ANSWER_PROMPT.format(context=state["context"])),
                             *history])
        return {"draft": draft}
    except LLMError as exc:
        logger.error("Generation failed: %s", exc)
        return {"draft": "", "error": str(exc)}


def quality_check_node(state: AgentState) -> dict:
    """Pass-through node: the decision is made by route_after_quality."""
    return {}


def respond_node(state: AgentState) -> dict:
    return {"messages": [AIMessage(content=state["draft"])], "escalated": False}


def escalate_node(state: AgentState) -> dict:
    """Hand the case to a human and create a ticket ID."""
    ticket = f"TKT-{uuid.uuid4().hex[:6].upper()}"
    if state.get("error"):
        reason = "I'm having technical trouble answering right now."
    elif state.get("angry"):
        reason = "I'm sorry for the frustration this has caused."
    else:
        reason = "I want to make sure you get the right answer on this."
    msg = (f"{reason} I've passed your case to a human agent. Your ticket ID is **{ticket}**, "
           f"and they will contact you within 24 hours.")
    return {"messages": [AIMessage(content=msg)], "escalated": True, "ticket_id": ticket}


# ---- routing functions (conditional edges) ----
def route_after_classify(state: AgentState) -> str:
    if state["category"] == "escalate" or state.get("angry"):
        return "escalate"
    return "order" if state["category"] == "order_status" else "kb"


def route_after_quality(state: AgentState) -> str:
    draft = state.get("draft", "")
    if state.get("error") or not draft.strip() or SENTINEL in draft or len(draft.strip()) < 10:
        return "escalate"
    return "respond"
