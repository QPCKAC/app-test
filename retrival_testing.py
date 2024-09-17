import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import base64
import fitz  # PyMuPDF

# Define directories
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_acog_docs_with_metadata")

# Initialize Ollama components
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load the existing Chroma database
@st.cache_resource
def load_vectordb():
    return Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

vectorstore = load_vectordb()

# Generate PDF link function
def generate_pdf_link(metadata):
    pdf_dir = os.path.join(current_dir, "data")
    pdf_path = os.path.join(pdf_dir, metadata['source'])
    return pdf_path, metadata['page']

# Display PDF function
def display_pdf(pdf_path, page, highlight_text=None):
    doc = fitz.open(pdf_path)
    images = []
    highlight_index = None
    for i in range(len(doc)):
        page_obj = doc.load_page(i)
        if i == page - 1 and highlight_text:
            text_instances = page_obj.search_for(highlight_text)
            if text_instances:
                highlight_index = i
                for inst in text_instances:
                    highlight = page_obj.add_highlight_annot(inst)
        pix = page_obj.get_pixmap()
        img_bytes = pix.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode()
        images.append(f'<img id="page-{i}" src="data:image/png;base64,{img_base64}" style="width:100%; margin-bottom:10px;"/>')
    
    scroll_script = ""
    if highlight_index is not None:
        scroll_script = f"""
        <script>
            document.addEventListener('DOMContentLoaded', (event) => {{
                document.getElementById('page-{highlight_index}').scrollIntoView({{behavior: 'smooth'}});
            }});
        </script>
        """
    
    pdf_display = f"""
    <div id="pdf-viewer" style="height:600px; overflow-y:scroll;">
        {"".join(images)}
    </div>
    {scroll_script}
    """
    st.components.v1.html(pdf_display, height=620, scrolling=True)
    doc.close()

# Show PDF function
def show_pdf(pdf_path, page, highlight_text=None):
    st.session_state.pdf_viewer = {
        "pdf_path": pdf_path,
        "page": page,
        "highlight_text": highlight_text
    }

# Create a retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5,
        "score_threshold": 0.6
    }
)

# Streamlit UI
st.title("ACOG Document Retrieval System")

# Initialize session state variables
if 'pdf_viewer' not in st.session_state:
    st.session_state.pdf_viewer = None
if 'docs' not in st.session_state:
    st.session_state.docs = None

# User input
query = st.text_input("Enter your question about ACOG guidelines:")

if query and (query != st.session_state.get('last_query', '')):
    st.session_state.last_query = query
    # Use invoke instead of get_relevant_documents
    st.session_state.docs = retriever.invoke(query)

if st.session_state.docs:
    # Display retrieved chunks
    st.markdown("### Retrieved Chunks:")
    for i, doc in enumerate(st.session_state.docs):
        with st.expander(f"Chunk {i+1}: {doc.metadata.get('source', 'Unknown')}"):
            st.write("Content:")
            st.write(doc.page_content)
            st.write(f"Source: {doc.metadata.get('source', 'Unknown')}")
            st.write(f"Page: {doc.metadata.get('page', 'Unknown')}")
            pdf_path, page = generate_pdf_link(doc.metadata)
            if st.button(f"View PDF (Page {page})", key=f"pdf_button_{i}"):
                show_pdf(pdf_path, page, doc.page_content[:50])  # Use the first 50 characters as highlight text
    
    # Display PDF if requested
    if st.session_state.pdf_viewer:
        st.markdown("### PDF Viewer")
        display_pdf(
            st.session_state.pdf_viewer["pdf_path"],
            st.session_state.pdf_viewer["page"],
            st.session_state.pdf_viewer["highlight_text"]
        )

# Add a sidebar with some information
st.sidebar.title("About")
st.sidebar.info("This app retrieves relevant chunks from ACOG guidelines using a local Chroma database.")

# Debug information
st.sidebar.title("Debug Info")
st.sidebar.write(f"Query: {st.session_state.get('last_query', 'No query yet')}")
st.sidebar.write(f"Number of docs retrieved: {len(st.session_state.docs) if st.session_state.docs else 0}")


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8501))
#     import streamlit.web.bootstrap as bootstrap
#     bootstrap.run(
#         __file__,
#         f"--server.port={port}",
#         "--server.address=0.0.0.0"
#     )