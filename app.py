import streamlit as st
import faiss
import os
from io import BytesIO
from docx import Document
import numpy as np

from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# ---------------------------------------
# PROCESS INPUT DOCUMENTS
# ---------------------------------------
def process_input(input_type, input_data):

    loader = None
    if input_type == "Link":
        loader = WebBaseLoader(input_data)
        documents = loader.load()

    elif input_type == "PDF":
        if input_data is not None:
            pdf_reader = PdfReader(input_data)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            documents = text
        else:
            raise ValueError("No PDF uploaded")

    elif input_type == "Text":
        if isinstance(input_data, str):
            documents = input_data
        else:
            raise ValueError("Expected a string for 'Text' input type.")

    elif input_type == "DOCX":
        if input_data is not None:
            doc = Document(input_data)
            text = "\n".join([para.text for para in doc.paragraphs])
            documents = text
        else:
            raise ValueError("No DOCX uploaded")

    elif input_type == "TXT":
        if input_data is not None:
            text = input_data.read().decode("utf-8")
            documents = text
        else:
            raise ValueError("No TXT uploaded")

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    if input_type == "Link":
        docs = text_splitter.split_documents(documents)
        texts = [str(doc.page_content) for doc in docs]
    else:
        texts = text_splitter.split_text(documents)

    # Embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Create FAISS index
    sample_embedding = np.array(hf_embeddings.embed_query("sample text"))
    dimension = sample_embedding.shape[0]
    index = faiss.IndexFlatL2(dimension)

    vector_store = FAISS(
        embedding_function=hf_embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Add documents
    vector_store.add_texts(texts)
    return vector_store


# ---------------------------------------
# ANSWER QUESTION USING LATEST LANGCHAIN API
# ---------------------------------------
def answer_question(vectorstore, query):

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="conversational",
        huggingfacehub_api_token=st.secrets["huggingface_api_key"],
        temperature=0.6
    )

    chat_llm = ChatHuggingFace(llm=llm)

    retriever = vectorstore.as_retriever()

    # New Retrieval chain system (Replaces RetrievalQA)
    document_chain = create_stuff_documents_chain(chat_llm)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    response = qa_chain.invoke({"input": query})
    return response["output_text"]


# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------
def main():

    st.markdown(
        "<h1 style='text-align: center; color: white;'>RAG Based Q&A APP</h1>",
        unsafe_allow_html=True
    )

    input_type = st.selectbox("Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])

    if input_type == "Link":
        number_input = st.number_input(min_value=1, max_value=20, step=1, label="Enter the number of Links")
        input_data = []
        for i in range(number_input):
            url = st.text_input(f"URL {i+1}")
            input_data.append(url)

    elif input_type == "Text":
        input_data = st.text_input("Enter the text")

    elif input_type == "PDF":
        input_data = st.file_uploader("Upload a PDF file", type=["pdf"])

    elif input_type == "TXT":
        input_data = st.file_uploader("Upload a text file", type=["txt"])

    elif input_type == "DOCX":
        input_data = st.file_uploader("Upload a DOCX file", type=["docx", "doc"])

    if st.button("Proceed"):
        vectorstore = process_input(input_type, input_data)
        st.session_state["vectorstore"] = vectorstore

    if "vectorstore" in st.session_state:
        query = st.text_input("Ask your question")
        if st.button("Submit"):
            answer = answer_question(st.session_state["vectorstore"], query)
            st.write(answer)


if __name__ == "__main__":
    main()
