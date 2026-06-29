# Intelligent RAG Chatbot using LangGraph

An AI-powered question-answering system that retrieves information from user-uploaded
documents (PDF / Markdown / TXT) and generates grounded answers using an LLM, instead of
relying on the model's general knowledge. Orchestrated with **LangGraph**, served through
a **Streamlit** chat UI.

**This version runs entirely on free tools:**
- **Embeddings** run locally on your laptop via `sentence-transformers` — no API key, no cost.
- **LLM (answer generation)** uses **Groq**, which has a free API tier with no credit card required.

## Architecture

```
Upload Documents → Document Processing → Chunking → Embedding Generation
→ Vector DB Indexing (Chroma)
─────────────────────────────────────────────────────────────────
User Question → Retrieve Relevant Chunks (LangGraph "retrieve" node)
→ LLM Generates Answer (LangGraph "generate" node) → Display Response
```

## Project Structure

```
rag_chatbot/
├── app.py                     # Streamlit UI
├── requirements.txt
├── .env.example                # copy to .env and add your API key
├── src/
│   ├── document_processor.py  # loading + chunking (PDF/MD/TXT)
│   ├── vectorstore.py         # embeddings + Chroma indexing/retrieval
│   └── graph.py                # LangGraph workflow (retrieve → generate)
├── sample_data/
│   ├── employee_handbook.md   # sample knowledge base file
│   └── faq.md                  # sample knowledge base file
└── vector_store/               # Chroma's persisted index (auto-created)
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Add your free Groq API key:
   ```bash
   cp .env.example .env
   ```
   Then go to [console.groq.com/keys](https://console.groq.com/keys), sign up (no credit card
   required), create a key, and paste it into `.env`:
   ```
   GROQ_API_KEY=gsk-...
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Using the Sample Data

1. Open the app in your browser (Streamlit prints a local URL).
2. In the sidebar, upload `sample_data/employee_handbook.md` and `sample_data/faq.md`.
3. Click **"Process & Index Documents."**
4. Ask questions like:
   - "How many casual leaves do I get per year?"
   - "What is the notice period during probation?"
   - "Can I work from home full time?"
   - "What happens if I damage my laptop?"

Each answer is generated **only** from the retrieved chunks. If you ask something unrelated
to the documents (e.g. "What's the capital of France?"), the bot will say it couldn't find
that information in the uploaded documents — this is the anti-hallucination guardrail built
into the system prompt.

## How It Works

- **document_processor.py** loads each file with the right LangChain loader (`PyPDFLoader`,
  `UnstructuredMarkdownLoader`, `TextLoader`) and splits it into overlapping chunks with
  `RecursiveCharacterTextSplitter`.
- **vectorstore.py** embeds chunks locally using `HuggingFaceEmbeddings`
  (`sentence-transformers/all-MiniLM-L6-v2`, downloaded once and cached) and stores them in a
  local **Chroma** vector database. Re-indexing replaces the previous collection so the bot
  only answers from the latest upload.
- **graph.py** defines a 2-node **LangGraph** `StateGraph`:
  - `retrieve`: semantic search over Chroma for the top-k relevant chunks.
  - `generate`: passes those chunks + chat history to **Groq's** Llama model with a strict
    "answer only from context" system prompt.
- **app.py** wires it all into a Streamlit chat interface, with an optional expander showing
  exactly which chunks were used for each answer (for transparency/debugging).

## Customization

- Swap models via `.env` (`EMBEDDING_MODEL`, `CHAT_MODEL`) without touching code.
- Adjust chunk size/overlap and retrieval `k` directly from the sidebar sliders.
- To support more file types, add a loader branch in `document_processor.load_document()`.
- To use a different vector DB (e.g. FAISS, Pinecone, Weaviate), swap the implementation in
  `vectorstore.py` — `graph.py` and `app.py` don't need to change since they only depend on
  the `similarity_search` interface.

## Notes

- This uses local sentence-transformers embeddings (free, offline after first download) and
  Groq for chat completions (free tier, no credit card). Swap `ChatGroq` in `graph.py` for
  another provider (OpenAI, Anthropic, Ollama) if you later want to pay for a larger model.
- The Chroma index is persisted to `vector_store/` so it survives app restarts until you
  re-index with new documents.
- The first time you run the app, it will download the embedding model (~90MB) from Hugging
  Face — this requires internet access once; after that it runs fully offline for embeddings.
