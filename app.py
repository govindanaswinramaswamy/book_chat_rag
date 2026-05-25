import os
import re
import json
from datetime import datetime

import streamlit as st

from src.engine import query_engine
from langchain_core.messages import HumanMessage, AIMessage


# config path
CONFIG_FILE_PATH = "config/config.yaml"

# create traces directory (if it does not exist)
TRACE_DIR = "traces"
os.makedirs(TRACE_DIR, exist_ok=True)

# -----------------------------------------
# page configuration
# -----------------------------------------

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
)

# -----------------------------------------
# title + intro
# -----------------------------------------

st.title("RAG Chatbot")

st.markdown(
    """
    <div style="color: gray; font-size: 0.9rem; line-height: 1.6;">
    Ask questions from the book <i>Designing Machine Learning Systems:
    An Iterative Process for Production</i>.<br>
    The chatbot retrieves relevant answers from the book and answers only using that context.<br><br>
    Sample questions:<br>
    - What renewed the interest of Machine Learning?<br>
    - What is stratified sampling and why is it used?<br>
    - What are the four key requirements of a production ML system?
    </div> <br>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------
# initialize session state
# -----------------------------------------

# stores LangChain chat messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# create one trace file per conversation session
if "trace_file_path" not in st.session_state:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.trace_file_path = os.path.join(
        TRACE_DIR,
        f"conversation_{timestamp}.jsonl"
    )

# -----------------------------------------
# display previous Q&A
# -----------------------------------------

for message in st.session_state.chat_history:

    # human messages are shown as user messages
    if isinstance(message, HumanMessage):
        role = "user"

    # AI messages are shown as assistant messages
    elif isinstance(message, AIMessage):
        role = "assistant"

    # skip unknown message types
    else:
        continue

    # render chat bubble
    with st.chat_message(role):
        st.markdown(message.content)


# -----------------------------------------
# chat input
# -----------------------------------------

question = st.chat_input("Ask a question...")


# -----------------------------------------
# run query engine
# -----------------------------------------

if question:

    # show user message immediately
    with st.chat_message("user"):
        st.markdown(question)

    # generate and show assistant response
    with st.chat_message("assistant"):

        with st.spinner("Retrieving answer..."):

            # call your RAG pipeline
            result = query_engine(
                question=question,
                config_file_path=CONFIG_FILE_PATH,
                chat_history=st.session_state.chat_history,
            )

            # save full trace to disk
            trace = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "question": question,
                "retrieval_question": result.get("retrieval_question"),
                "context": result.get("context"),
                "answer": result.get("answer"),
                "reason": result.get("reason"),
                "rag_error": result.get("rag_error"),
            }
            with open(st.session_state.trace_file_path, "a", encoding="utf-8") as f:
                # write one JSON object per line
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")

            # safely extract answer
            answer = result.get(
                "answer",
                "I could not generate an answer.",
            )

            # convert citations:
            # [c. 1, p. 23] -> [p. 23]
            answer = re.sub(
                r"\[c\.\s*\d+\s*,\s*p\.\s*(\d+)\]",
                r"[p. \1]",
                answer,
            )

            # display answer
            st.markdown(answer)

    # save conversation as LangChain messages
    st.session_state.chat_history.extend(
        [
            HumanMessage(content=question),
            AIMessage(content=answer),
        ]
    )