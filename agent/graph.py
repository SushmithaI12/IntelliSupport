"""LangGraph workflow: classify -> (order tool | retrieve) -> generate -> quality check -> respond/escalate."""
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (classify_node, escalate_node, generate_node, order_lookup_node,
                         quality_check_node, respond_node, retrieve_node,
                         route_after_classify, route_after_quality)
from agent.state import AgentState


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify", classify_node)
    g.add_node("order_lookup", order_lookup_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("quality_check", quality_check_node)
    g.add_node("respond", respond_node)
    g.add_node("escalate", escalate_node)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_after_classify,
                            {"escalate": "escalate", "order": "order_lookup", "kb": "retrieve"})
    g.add_edge("order_lookup", "generate")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "quality_check")
    g.add_conditional_edges("quality_check", route_after_quality,
                            {"respond": "respond", "escalate": "escalate"})
    g.add_edge("respond", END)
    g.add_edge("escalate", END)
    return g.compile(checkpointer=MemorySaver())  # MemorySaver = per-thread conversation memory


def run_agent(graph, user_text: str, thread_id: str) -> dict:
    """Send one message; the thread_id keeps the conversation memory separate per chat."""
    result = graph.invoke({"messages": [HumanMessage(content=user_text)]},
                          config={"configurable": {"thread_id": thread_id}})
    return {"reply": result["messages"][-1].content,
            "category": result.get("category", "general"),
            "confidence": result.get("confidence", 0.0),
            "escalated": result.get("escalated", False),
            "ticket_id": result.get("ticket_id")}
