from pathlib import Path
from llama_index.readers.file import PDFReader, PyMuPDFReader, RTFReader
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
import os
from llama_index.retrievers.bm25 import BM25Retriever

from pinecone import Pinecone, Index, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
# Load environment variables from .env file
load_dotenv()



# # [Optional] Delete the index before re-running the tutorial.
# # pinecone.delete_index(index_name)
pc = Pinecone()
index_name = "llamaindex-risk-ld"

try:
    if index_name not in pc.list_indexes().names():
        print(f"Creating new index: {index_name}")
        pc.create_index(
            index_name,
            dimension=1536,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"Index {index_name} already exists")

    pinecone_index = pc.Index(index_name)

    # Attempt to delete all contents in the index
    try:
        print("Attempting to delete all contents in the index...")
        pinecone_index.delete(deleteAll=True)
        print("Successfully deleted all contents in the index")
    except Exception as e:
        print(f"Error deleting index contents: {str(e)}")
        print("Continuing with the existing index...")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

except Exception as e:
    print(f"An error occurred while setting up Pinecone: {str(e)}")
    print(f"Error type: {type(e).__name__}")



### Simple Directory Reader
# documents = SimpleDirectoryReader("./RTFs", filename_as_id=True).load_data()
# print([x.doc_id for x in documents])


### PDF Reader

# loader = PDFReader()
# documents = loader.load_data(file=Path('./PDFs/Intrapartum Management of Intraamniotic Infection.pdf'))

# parser = PyMuPDFReader()
# file_extractor = {".pdf": parser}
# documents = SimpleDirectoryReader(
#     "./PDFs", file_extractor=file_extractor, filename_as_id=True
# ).load_data()


# print([x.doc_id for x in documents])
# print(documents[0])

###RTF
parser = RTFReader()
file_extractor = {".rtf": parser}
documents = SimpleDirectoryReader(
    "./data",required_exts=[".rtf"], file_extractor=file_extractor, filename_as_id=True
).load_data()



# print([x.doc_id for x in documents])
# print(documents[0].text[:100])  # Print first 100 characters of the first document

### Define Semantic Splitter
# embed_model = OpenAIEmbedding()
# # also baseline splitter
# base_splitter = SentenceSplitter(chunk_size=512)
# splitter = SemanticSplitterNodeParser(
#     buffer_size=1, breakpoint_percentile_threshold=90, embed_model=embed_model, include_metadata=True, sentence_splitter=base_splitter
# )



# nodes = splitter.get_nodes_from_documents(documents)
# print(nodes[0].get_content(metadata_mode="all"))


from llama_index.core.node_parser import SentenceSplitter
text_parser = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
)
text_chunks = []
# maintain relationship with source doc index, to help inject doc metadata in (3)
doc_idxs = []
for doc_idx, page in enumerate(documents):
    page_text = page.get_text()
    cur_text_chunks = text_parser.split_text(page_text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))


from llama_index.core.schema import TextNode
nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc_idx = doc_idxs[idx]
    src_page = documents[src_doc_idx]
    node.metadata = src_page.metadata
    nodes.append(node)

print(nodes[0].metadata)
# print a sample node
print(nodes[0].get_content(metadata_mode="all"))

# from llama_index.core.extractors import (
#     QuestionsAnsweredExtractor,
#     TitleExtractor,
# )
# from llama_index.core.ingestion import IngestionPipeline
# from llama_index.llms.openai import OpenAI

# llm = OpenAI(model="gpt-4o")

# extractors = [
#     TitleExtractor(nodes=5, llm=llm),
#     QuestionsAnsweredExtractor(questions=3, llm=llm),
# ]

# pipeline = IngestionPipeline(
#     transformations=extractors,
# )

# async def process_nodes():
#     return await pipeline.arun(nodes=nodes, in_place=False)

# nodes = asyncio.run(process_nodes())

# print(nodes[0].metadata)

from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding()
for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding


vector_store.add(nodes)