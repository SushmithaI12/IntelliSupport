"""Typed state shared by every node in the LangGraph workflow."""
from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # full conversation (memory)
    category: str          # billing | technical | order_status | general | escalate
    confidence: float      # classifier confidence, 0 to 1
    angry: bool            # customer sentiment flag
    context: str           # knowledge-base text or order data used for the answer
    draft: str             # answer waiting for the quality check
    escalated: bool        # True when handed to a human
    ticket_id: Optional[str]
    error: Optional[str]   # set when the LLM failed after retries
