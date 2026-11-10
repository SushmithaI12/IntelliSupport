# 🎧 SupportPilot-Agent

AI customer support agent built with **LangGraph**, **LangChain** and the **Groq LLM API**. It classifies each query, answers from a knowledge base, remembers the conversation, calls an order-lookup tool, and escalates to a human when it can't help.

## Features
- **Query classification**: billing, technical, order status, general or escalate, with a confidence score and sentiment flag
- **Multi-step LangGraph workflow** with conditional routing
- **Conversational memory** using LangGraph `MemorySaver` and a `thread_id` per chat
- **Tool use**: mock order lookup (`ORD-1001`, `ORD-1002`, `ORD-1003`)
- **Grounded answers**: the model answers only from retrieved FAQ context
- **Escalation** with a ticket ID when the customer is angry, asks for a human, the topic is unknown, or the answer fails the quality check
- **Failure handling**: retries (2), empty-response checks, keyword fallback classifier and a safe handoff message instead of a crash
- **Streamlit UI** with a sidebar showing category, confidence and escalation status

## Architecture
```
START -> classify -+-> escalate ------------------------------> END
                   |
                   +-> order_lookup --+
                   |                  +-> generate -> quality_check -+-> respond  -> END
                   +-> retrieve ------+                              |
                                                                     +-> escalate -> END
```

## Project structure
```
app.py                  Streamlit chat UI
agent/state.py          Typed graph state
agent/nodes.py          Node functions, LLM retry helper, routing logic
agent/graph.py          LangGraph workflow and run_agent()
agent/knowledge_base.py FAQ data, keyword retrieval, mock order tool
tests/test_flow.py      Offline tests with a fake LLM (no API key needed)
```

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
# add your Groq API key and model name in .env
python -m streamlit run app.py
```
Run the tests (no API key needed):
```bash
python tests/test_flow.py
```

## Example conversations
| Customer says | Agent behaviour |
|---|---|
| "How do I get a refund?" | Classified as billing, answers from the refund FAQ |
| "Where is my order ORD-1001?" | Order tool returns status, carrier and expected date |
| "Can you check it again?" (next message) | Memory finds ORD-1001 from earlier in the chat |
| "This is the worst service!" | Detected as angry, escalated with a ticket ID |
| "Recommend a pizza place" | Not in the knowledge base, escalated safely |

## Tech stack
Python, LangChain, LangGraph, Groq API, Streamlit, python-dotenv

## Credits
Project idea from the "5 AI Projects for 2026" list; architecture and code written by me.
