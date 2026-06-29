"""
app.py
------
Streamlit UI for the Intelligent RAG Chatbot.

Workflow:
  1. User uploads PDF / Markdown / TXT documents.
  2. Documents are chunked, embedded, and indexed in Chroma.
  3. User asks questions in a chat box.
  4. LangGraph retrieves relevant chunks and generates a grounded answer.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.document_processor import process_files
from src.vectorstore import build_vectorstore
from src.graph import build_graph, run_query

load_dotenv()

st.set_page_config(page_title="Intelligent RAG Chatbot", page_icon="🤖", layout="wide")

# ---------- Session state ----------
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# ---------- Sidebar: document upload & indexing ----------
with st.sidebar:
    st.header("📄 Knowledge Base")

    if not os.getenv("GROQ_API_KEY"):
        st.warning("Set GROQ_API_KEY in your .env file before chatting (get a free key at console.groq.com/keys).")

    uploaded_files = st.file_uploader(
        "Upload PDF, Markdown, or TXT files",
        type=["pdf", "md", "markdown", "txt"],
        accept_multiple_files=True,
    )

    chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, 4)

    if st.button("Process & Index Documents", type="primary", disabled=not uploaded_files):
        with st.spinner("Processing documents..."):
            # Save uploads to a temp directory so loaders can read real file paths
            tmp_dir = tempfile.mkdtemp()
            saved_paths = []
            for f in uploaded_files:
                path = os.path.join(tmp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                saved_paths.append(path)

            chunks = process_files(saved_paths)
            if not chunks:
                st.error("No content could be extracted from the uploaded files.")
            else:
                vectordb = build_vectorstore(chunks)
                st.session_state.vectordb = vectordb
                st.session_state.graph = build_graph(vectordb, k=top_k)
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.session_state.messages = []
                st.success(f"Indexed {len(chunks)} chunks from {len(saved_paths)} file(s).")

    if st.session_state.indexed_files:
        st.markdown("**Indexed files:**")
        for name in st.session_state.indexed_files:
            st.markdown(f"- {name}")

    show_sources = st.checkbox("Show retrieved chunks with each answer", value=True)

# ---------- Main chat area ----------
st.title("🤖 Intelligent RAG Chatbot")
st.caption("Ask questions about your uploaded documents. Answers are grounded in retrieved content only.")

if st.session_state.graph is None:
    st.info("Upload and index documents in the sidebar to start chatting.")
else:
    # Render existing chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Retrieved chunks used for this answer"):
                    for doc in msg["sources"]:
                        st.markdown(f"**Source:** {doc.metadata.get('source', 'unknown')}")
                        st.markdown(doc.page_content[:800])
                        st.markdown("---")

    question = st.chat_input("Ask a question about your documents...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Build chat history as (role, content) tuples for the prompt
        history = [
            (m["role"], m["content"])
            for m in st.session_state.messages[:-1]
            if m["role"] in ("user", "assistant")
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_query(st.session_state.graph, question, history)
                answer = result["answer"]
                sources = result["context"]
                st.markdown(answer)
                if show_sources and sources:
                    with st.expander("📚 Retrieved chunks used for this answer"):
                        for doc in sources:
                            st.markdown(f"**Source:** {doc.metadata.get('source', 'unknown')}")
                            st.markdown(doc.page_content[:800])
                            st.markdown("---")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
