import os
import sys
from typing import List, Dict
import fitz  # PyMuPDF
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
except ImportError:
    print("Error: Required packages are not installed.")
    print("Please run: pip install langchain langchain_community langchain_openai chromadb pymupdf")
    sys.exit(1)

# Define directories
current_dir = os.path.dirname(os.path.abspath(__file__))
acog_dir = os.path.join(current_dir, "data")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_acog_docs_with_metadata")

print(f"ACOG directory: {acog_dir}")
print(f"Persistent directory: {persistent_directory}")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def extract_pdf_metadata(file_path: str) -> Dict[str, str]:
    doc = fitz.open(file_path)
    metadata = doc.metadata
    doc.close()
    return metadata

def ingest_docs():
    if not os.path.exists(persistent_directory):
        print("Persistent directory does not exist. Initializing vector store...")

        if not os.path.exists(acog_dir):
            raise FileNotFoundError(f"The directory {acog_dir} does not exist. Please check the path.")

        pdf_files = [f for f in os.listdir(acog_dir) if f.endswith(".pdf")]

        if not pdf_files:
            print("No PDF files found in the ACOG directory.")
            return

        documents = []
        for pdf_file in pdf_files:
            file_path = os.path.join(acog_dir, pdf_file)
            try:
                pdf_metadata = extract_pdf_metadata(file_path)
                doc = fitz.open(file_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    metadata = {
                        "source": pdf_file,
                        "page": page_num + 1,
                        "title": pdf_metadata.get("title", ""),
                        "author": pdf_metadata.get("author", ""),
                        "subject": pdf_metadata.get("subject", ""),
                        "keywords": pdf_metadata.get("keywords", ""),
                    }
                    documents.append({"page_content": text, "metadata": metadata})
                doc.close()
                print(f"Successfully loaded: {pdf_file}")
            except Exception as e:
                print(f"Error loading {pdf_file}: {str(e)}")

        if not documents:
            print("No documents were successfully loaded. Exiting.")
            return

        print(f"Loaded {len(documents)} pages from {len(pdf_files)} PDF files")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.create_documents([doc["page_content"] for doc in documents], metadatas=[doc["metadata"] for doc in documents])

        print(f"Going to add {len(split_documents)} chunks to Chroma")
        db = Chroma.from_documents(
            split_documents, 
            embeddings, 
            persist_directory=persistent_directory
        )
        db.persist()
        print("****Loading to vectorstore done ***")
    else:
        print("Vector store already exists. No need to initialize.")

if __name__ == "__main__":
    try:
        ingest_docs()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("If this is a dependency error, please ensure all required packages are installed.")
        print("Run: pip install langchain langchain_community langchain_openai chromadb pymupdf")