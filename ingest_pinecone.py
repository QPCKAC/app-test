import os
import sys
from typing import List, Dict
import fitz  # PyMuPDF
from langchain_ollama import OllamaEmbeddings
from pinecone import Pinecone, PodSpec
from dotenv import load_dotenv

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Pinecone as LangchainPinecone
except ImportError:
    print("Error: Required packages are not installed.")
    print("Please run: pip install langchain langchain_ollama pinecone-client pymupdf python-dotenv")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# Define directories
current_dir = os.path.dirname(os.path.abspath(__file__))
acog_dir = os.path.join(current_dir, "data")

print(f"ACOG directory: {acog_dir}")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "acog-docs")

# Initialize Nomic embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Check embedding dimension
test_embedding = embeddings.embed_query("Test sentence")
embedding_dim = len(test_embedding)
print(f"Embedding dimension: {embedding_dim}")

def extract_pdf_metadata(file_path: str) -> Dict[str, str]:
    doc = fitz.open(file_path)
    metadata = doc.metadata
    doc.close()
    return metadata

def ingest_docs():
    # Check if the index exists, if not create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric='cosine',
            spec=PodSpec(
                environment="gcp-starter"  # Use gcp-starter for free plan
            )
        )
        print(f"Created new Pinecone index: {index_name}")
    
    index = pc.Index(index_name)
    
    if index.describe_index_stats()['total_vector_count'] == 0:
        print("Pinecone index is empty. Initializing vector store...")

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
                        "type": "pdf"
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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        split_documents = text_splitter.create_documents([doc["page_content"] for doc in documents], metadatas=[doc["metadata"] for doc in documents])

        print(f"Going to add {len(split_documents)} chunks to Pinecone")
        LangchainPinecone.from_documents(split_documents, embeddings, index_name=index_name)
        print("****Loading to vectorstore done ***")
    else:
        print("Vector store already exists. No need to initialize.")

if __name__ == "__main__":
    try:
        ingest_docs()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("If this is a dependency error, please ensure all required packages are installed.")
        print("Run: pip install langchain langchain_ollama pinecone-client pymupdf python-dotenv")