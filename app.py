import streamlit as st
import faiss
import os
from io import BytesIO
from docx import Document
import numpy as np

from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader

# ✅ Old LangChain versions support this
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint


# ------------------------------------------
# Process input data
# ------------------------------------------
def process_input(input_type, input_data):

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
        documents = input_data

    elif input_type == "DOCX":
        doc = Document(input_data)
        documents = "\n".join([p.text for p in doc.paragraphs])

    elif input_type == "TXT":
        documents = input_data.read().decode("utf-8")

    # ✅ Split text correctly
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    if input_type == "Link":
        docs = text_splitter.split_documents(documents)
        texts = [str(doc.page_content) for doc in docs]
    else:
        texts = text_splitter.split_text(documents)

    # ✅ Embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    # ✅ Create FAISS index
    sample_vector = np.array(hf_embeddings.embed_query("hello"))
    dimension = sample_vector.shape[0]

    index = faiss.IndexFlatL2(dimension)

    vector_store = FAISS(
        embedding_function=hf_embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    vector_store.add_texts(texts)
    return vector_store


# ------------------------------------------
# Answer question using RetrievalQA
# ------------------------------------------
def answer_question(vectorstore, query):

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="conversational",
        huggingfacehub_api_token=st.secrets["huggingface_api_key"],
        temperature=0.6
    )

    chat_llm = ChatHuggingFace(llm=llm)

    # ✅ Works ONLY in LangChain <= 0.1.17
    qa = RetrievalQA.from_chain_type(
        llm=chat_llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    answer = qa({"query": query})
    return answer


# ------------------------------------------
# Streamlit UI
# ------------------------------------------
def main():

    st.markdown(
        "<h1 style='text-align:center; color:white;'>RAG Based Q&A APP</h1>",
        unsafe_allow_html=True
    )

    input_type = st.selectbox("Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])

    if input_type == "Link":
        count = st.number_input("Number of Links", min_value=1, max_value=20, step=1)
        input_data = []
        for i in range(count):
            url = st.text_input(f"URL {i+1}")
            input_data.append(url)

    elif input_type == "Text":
        input_data = st.text_input("Enter text")

    elif input_type == "PDF":
        input_data = st.file_uploader("Upload PDF", type=["pdf"])

    elif input_type == "TXT":
        input_data = st.file_uploader("Upload TXT", type=["txt"])

    elif input_type == "DOCX":
        input_data = st.file_uploader("Upload DOCX", type=["docx", "doc"])

    if st.button("Proceed"):
        vectorstore = process_input(input_type, input_data)
        st.session_state["vectorstore"] = vectorstore

    if "vectorstore" in st.session_state:
        query = st.text_input("Ask your question:")
        if st.button("Submit"):
            answer = answer_question(st.session_state["vectorstore"], query)
            st.write(answer["result"])


if __name__ == "__main__":
    main()
