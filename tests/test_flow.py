"""Offline tests: run the real graph with a fake LLM (no API key needed).
Run from the project root:  python tests/test_flow.py
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage  # noqa: E402

from agent import nodes  # noqa: E402
from agent.graph import build_graph, run_agent  # noqa: E402


class FakeLLM:
    fail = False

    def invoke(self, messages):
        if FakeLLM.fail:
            raise RuntimeError("API down")
        system = messages[0].content
        last = " ".join(m.content.lower() for m in messages[1:] if m.type == "human")
        if "classifier" in system:
            cat = "order_status" if "order" in last else "billing" if "refund" in last else "general"
            return AIMessage(content='{"category": "%s", "confidence": 0.9, "angry": false}' % cat)
        return AIMessage(content="Here is the answer based on: " + system.split("Context:")[1].strip()[:60])


nodes.get_llm = lambda: FakeLLM()
nodes.time.sleep = lambda s: None  # skip retry delay in tests
graph = build_graph()


def chat(text, tid):
    return run_agent(graph, text, tid)


def test_answer_from_kb():
    out = chat("How do I get a refund?", str(uuid.uuid4()))
    assert not out["escalated"] and out["category"] == "billing"


def test_order_tool():
    out = chat("Where is my order ORD-1001?", str(uuid.uuid4()))
    assert not out["escalated"] and "ORD-1001" in out["reply"]


def test_memory_remembers_order_id():
    tid = str(uuid.uuid4())
    chat("I want to check my order ORD-1002", tid)
    out = chat("Can you check it again?", tid)
    assert "ORD-1002" in out["reply"]


def test_human_request_escalates():
    out = chat("Let me talk to a human please", str(uuid.uuid4()))
    assert out["escalated"] and out["ticket_id"].startswith("TKT-")


def test_angry_customer_escalates():
    out = chat("This is the worst service, I am furious about my refund", str(uuid.uuid4()))
    assert out["escalated"]


def test_unknown_topic_escalates():
    out = chat("Can you recommend a good pizza place?", str(uuid.uuid4()))
    assert out["escalated"]


def test_llm_failure_escalates_safely():
    FakeLLM.fail = True
    try:
        out = chat("How do I reset my password?", str(uuid.uuid4()))
    finally:
        FakeLLM.fail = False
    assert out["escalated"] and "technical trouble" in out["reply"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} tests passed.")
