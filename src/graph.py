"""
graph.py
--------
Defines the LangGraph workflow that orchestrates the RAG pipeline:

    START -> retrieve -> generate -> END

The graph receives a question (and optional chat history), retrieves
relevant chunks from the vector database, and asks the LLM to answer
using ONLY that retrieved context.
"""

import os
from typing import List,TypedDict
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END

CHAT_MODEL=os.getenv("CHAT_MODEL","llama-3.1-8b-instant")

SYSTEM_PROMPT=""" You are a helpful assistant that answers questions using ONLY the provided document context.
Follow below rules strictly:

    1.Answer only using information that found in the context below
    2.If the answer is not present in the context, say:
        "I couldn't find this information in the uploaded documents.        
    3.Be concise and accurate. Do not invent facts.     
    4.If helpful, mention which source document the answer came from.
    
Context:
{context}
"""

class GraphState(TypedDict):
    question:str
    chat_history:List[tuple]
    context:List[Document]
    answer:str

def _format_context(docs:List[Document]) -> str:
    if not docs:
        return "No relevant context was found."
    parts=[]
    for doc in docs:
        source=doc.metadata["source"] if "source" in doc.metadata else "unknown"
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_graph(vectordb, k: int=4):
    """
    Build and compile the LangGraph workflow.
    `vectordb` is an already-built Chroma vector store for the uploaded docs.
    """

    llm=ChatGroq(model=CHAT_MODEL,temperature=0)

    def retrieve_node(state:GraphState)->GraphState:
        docs=vectordb.similarity_search(state["question"],k=k)
        return {"context":docs}
    
    def generate_node(state:GraphState)->GraphState:
        context_text=_format_context(state["context"])
        system_content=SYSTEM_PROMPT.format(context=context_text)

        messages=[SystemMessage(content=system_content)]
        messages.extend(state.get("chat_history",[]))
        messages.append(HumanMessage(content=state["question"]))

        response=llm.invoke(messages)
        return {"answer":response.content}
    
    graph=StateGraph(GraphState)

    graph.add_node("retrieve",retrieve_node)
    graph.add_node("generate",generate_node)

    graph.add_edge(START,"retrieve")
    graph.add_edge("retrieve","generate")
    graph.add_edge("generate",END)

    return graph.compile()

def run_query(graph, question: str, chat_history: List[tuple]=None) -> GraphState:
    """Run the compiled graph for a single question and return only the answer."""
    result = graph.invoke(
            {
                "question": question,
                "chat_history": chat_history or [],
                "context": [],
                "answer": "",
            }
    )
    return result