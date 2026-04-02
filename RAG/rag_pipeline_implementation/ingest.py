
import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")




# db path and knowledge base path
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


#Return langchain documents from knowledge base 
def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", 
                                 loader_cls=TextLoader, 
                                 loader_kwargs={'encoding': 'utf-8'})
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


# Create chunks from documents
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

# Create embeddings and store in ChromaDB
def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(
            persist_directory=DB_NAME, 
            embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_NAME)
    
    collection = vectorstore._collection
    count = collection.count()
    
    sample_embedding = collection.get(limit=1,include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"Created embeddings for {count} chunks with dimension {dimensions}")
    
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete!")