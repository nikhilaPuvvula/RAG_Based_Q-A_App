import streamlit as st
import faiss
import os
from docx import Document
import numpy as np
from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Get API key from environment variable 
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

# ---------------------------
# Function to process inputs
# ---------------------------
def process_input(input_type, input_data):
    loader = None
    documents = None

    if input_type == "Link":
        documents = []
        for url in input_data:
            if url.strip():
                loader = WebBaseLoader(url)
                documents.extend(loader.load())

    elif input_type == "PDF":
        if input_data is not None:
            pdf_reader = PdfReader(input_data)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            documents = text

    elif input_type == "Text":
        documents = input_data

    elif input_type == "DOCX":
        if input_data is not None:
            doc = Document(input_data)
            text = "\n".join([para.text for para in doc.paragraphs])
            documents = text

    elif input_type == "TXT":
        if input_data is not None:
            text = input_data.read().decode("utf-8")
            documents = text

    if not documents:
        raise ValueError("No valid documents found!")

    # Splitting into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    if input_type == "Link":
        texts = text_splitter.split_documents(documents)
        texts = [str(doc.page_content) for doc in texts]
    else:
        texts = text_splitter.split_text(documents)

    # Embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    # FAISS index
    sample_embedding = np.array(hf_embeddings.embed_query("sample text"))
    dimension = sample_embedding.shape[0]
    index = faiss.IndexFlatL2(dimension)

    vector_store = FAISS(
        embedding_function=hf_embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    vector_store.add_texts(texts)
    return vector_store


# ---------------------------
# Function to answer queries
# ---------------------------
def answer_question(vectorstore, query):
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="conversational",
        huggingfacehub_api_token=huggingface_api_key,
        temperature=0.6,
    )

    chat_llm = ChatHuggingFace(llm=llm)
    qa = RetrievalQA.from_chain_type(
        llm=chat_llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    answer = qa({"query": query})
    return answer


# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.markdown(
        "<h1 style='text-align: center; color: white;'>🤖 RAG Based Q&A APP</h1>",
        unsafe_allow_html=True,
    )

    input_type = st.selectbox(
        "Choose Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"]
    )

    input_data = None
    if input_type == "Link":
        num_links = st.number_input(
            "Number of Links", min_value=1, max_value=10, value=1, step=1
        )
        input_data = []
        for i in range(num_links):
            url = st.text_input(f"Enter URL {i+1}")
            if url:
                input_data.append(url)

    elif input_type == "Text":
        input_data = st.text_area("Enter the text")

    elif input_type == "PDF":
        input_data = st.file_uploader("Upload a PDF", type=["pdf"])

    elif input_type == "TXT":
        input_data = st.file_uploader("Upload a TXT file", type=["txt"])

    elif input_type == "DOCX":
        input_data = st.file_uploader("Upload a DOCX file", type=["docx", "doc"])

    if st.button("Process Document"):
        try:
            vectorstore = process_input(input_type, input_data)
            st.session_state["vectorstore"] = vectorstore
            st.success(" Document processed successfully!")
        except Exception as e:
            st.error(f"Error: {e}")

    if "vectorstore" in st.session_state:
        query = st.text_input("Ask a question")
        if st.button("Submit"):
            with st.spinner("Generating answer..."):
                answer = answer_question(st.session_state["vectorstore"], query)
                st.markdown(f"**Answer:** {answer['result']}")


if __name__ == "__main__":
    main()
