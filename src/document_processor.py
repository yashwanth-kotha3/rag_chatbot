"""
document_processor.py
----------------------
Handles document ingestion: loading raw files (PDF, Markdown, TXT)
and splitting them into overlapping chunks suitable for embedding.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import(PyPDFLoader,TextLoader,UnstructuredMarkdownLoader)


def load_document(file_path:str)->List[Document]:
    """
    Load a single file into a list of LangChain Document objects,
    choosing the loader based on file extension.
    """
    ext=os.path.splitext(file_path)[1].lower()

    if ext==".pdf":
        Loader=PyPDFLoader(file_path)
    elif ext==".txt":
        Loader=TextLoader(file_path,encoding="utf-8")
    elif ext==".md" or ext==".markdown":
        Loader=UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")    
    
    documents=Loader.load()

    #attaching a clean source name to document.metadata
    for doc in documents:
        doc.metadata["source"]=os.path.basename(file_path)
    
    return documents
    
    
def load_documents(file_paths:List[str])->List[Document]:
    """Load multiple files and return a combined list of Documents."""
    all_docs:List[Document]=[]
    for path in file_paths:
        try:
            all_docs.extend(load_document(path))
        except Exception as e:
            print(f"[document_processor] Failed to load {path}:{e}")
    return all_docs


def chunks_documents( documents:List[Document],chunk_size:int=1000,chunk_overlap:int=150)->List[Document]:
    """
    Split documents into overlapping chunks.
    chunk_overlap helps preserve context across chunk boundaries.
    """
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,separators=["\n\n","\n"," ",""])
    chunks=splitter.split_documents(documents)

    for i,chunk in enumerate(chunks):
        chunk.metadata["chunk_id"]=i

    return chunks

def process_files(file_paths:List[str])->List[Document]:
    """Convenience function: load + chunk in one call."""
    raw_docs=load_documents(file_paths)
    return chunks_documents(raw_docs)
