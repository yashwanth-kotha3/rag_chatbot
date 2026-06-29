"""
vectorstore.py
--------------
Wraps embedding generation and Chroma vector database operations:
indexing document chunks and retrieving the most relevant ones for a query.

Embeddings run LOCALLY on your machine via sentence-transformers
(no API key, no cost, works fully offline after the model downloads once).
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")

EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")
PERSIST_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),"vector_store")

_embeddings_cache=None
def get_embeddings()->HuggingFaceEmbeddings:
    """
    Load the local embedding model once and reuse it.
    The first call downloads the model (~90MB) from Hugging Face;
    subsequent calls/runs use the cached local copy.
    """
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings_cache


def build_vectorstore(chunks:List[Document],collection_name:str="rag_chatbot")->Chroma:
    """
    Embed chunks and index them in a fresh Chroma collection.
    Each new document upload creates/overwrites this collection so the
    chatbot only answers from the latest set of uploaded files.
    """
    embeddings=get_embeddings()

    vectordb=Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=PERSIST_DIR,
            )
    try:
        vectordb.delete_collection()
    except Exception:
        pass

    vectordb=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=PERSIST_DIR,
    ) 
    return vectordb





def load_vectorstore(collection_name: str = "rag_chatbot") -> Chroma:
    """Load an already-persisted Chroma collection (e.g. across reruns)."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def retrieve_relevant_chunks(vectordb: Chroma, query: str, k: int = 4) -> List[Document]:
    """Return the top-k most semantically similar chunks to the query."""
    return vectordb.similarity_search(query, k=k)
