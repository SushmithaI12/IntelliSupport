"""Streamlit chat UI for the SupportPilot support agent."""
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from agent.graph import build_graph, run_agent
from agent.guards import SMALLTALK, smalltalk_reply

load_dotenv()
st.set_page_config(page_title="SupportPilot Agent", page_icon="🎧", layout="wide")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Please add it to the .env file.")
    st.stop()


@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.meta = None

with st.sidebar:
    st.header("Agent insights")
    meta = st.session_state.meta
    if meta:
        st.metric("Detected category", meta["category"])
        st.metric("Confidence", f"{meta['confidence']:.0%}")
        if meta["escalated"]:
            st.warning(f"Escalated to human. Ticket: {meta['ticket_id']}")
        else:
            st.success("Resolved by agent")
    else:
        st.caption("Send a message to see how the agent classified it.")
    st.divider()
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.meta = None
        st.rerun()
    st.caption("Try order IDs: ORD-1001, ORD-1002, ORD-1003")

st.title("🎧 SupportPilot: AI Customer Support Agent")
st.write("Ask about refunds, shipping, passwords, billing or your order status.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("How can we help you today?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if SMALLTALK.match(question):
            reply = smalltalk_reply(question)
        else:
            with st.spinner("Thinking..."):
                try:
                    out = run_agent(graph, question, st.session_state.thread_id)
                    reply = out["reply"]
                    st.session_state.meta = out
                except Exception:
                    reply = "Sorry, something went wrong on our side. Please try again in a moment."
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()