import streamlit as st
import numpy as np
import faiss
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# ✅ New RAG chain system (replaces old RetrievalQA)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# ----------------------------------------------------------
# PROCESS INPUT DOCUMENTS
# ----------------------------------------------------------
def process_input(input_type, input_data):

    if input_type == "Link":
        loader = WebBaseLoader(input_data)
        documents = loader.load()
        text_list = [doc.page_content for doc in documents]

    elif input_type == "PDF":
        pdf_reader = PdfReader(input_data)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
        text_list = [text]

    elif input_type == "Text":
        text_list = [input_data]

    elif input_type == "DOCX":
        doc = Document(input_data)
        text = "\n".join(p.text for p in doc.paragraphs)
        text_list = [text]

    elif input_type == "TXT":
        text = input_data.read().decode("utf-8")
        text_list = [text]

    # ✅ Split text using new splitter package
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for txt in text_list:
        chunks.extend(splitter.split_text(txt))

    # ✅ Embeddings model
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )

    # ✅ Create FAISS index
    vec = embedder.embed_query("hello")
    dim = len(vec)
    index = faiss.IndexFlatL2(dim)

    vector_db = FAISS(
        embedding_function=embedder.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_db.add_texts(chunks)
    return vector_db


# ----------------------------------------------------------
# ANSWER QUESTION USING LATEST LANGCHAIN RAG PIPELINE
# ----------------------------------------------------------
def answer_question(vectorstore, query):

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="conversational",
        huggingfacehub_api_token=st.secrets["huggingface_api_key"],
        temperature=0.6
    )

    chat_llm = ChatHuggingFace(llm=llm)

    retriever = vectorstore.as_retriever()

    # ✅ New LangChain RAG chain
    doc_chain = create_stuff_documents_chain(chat_llm)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    res = rag_chain.invoke({"input": query})
    return res["output_text"]


# ----------------------------------------------------------
# STREAMLIT APP UI
# ----------------------------------------------------------
def main():

    st.markdown(
        "<h1 style='text-align:center; color:white;'>RAG Based Q&A APP</h1>",
        unsafe_allow_html=True
    )

    input_type = st.selectbox("Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])

    if input_type == "Link":
        input_data = st.text_input("Enter URL")

    elif input_type == "Text":
        input_data = st.text_area("Enter text")

    elif input_type == "PDF":
        input_data = st.file_uploader("Upload PDF", type=["pdf"])

    elif input_type == "DOCX":
        input_data = st.file_uploader("Upload DOCX", type=["docx", "doc"])

    elif input_type == "TXT":
        input_data = st.file_uploader("Upload TXT", type=["txt"])

    if st.button("Process"):
        vectorstore = process_input(input_type, input_data)
        st.session_state["vs"] = vectorstore

    if "vs" in st.session_state:
        query = st.text_input("Ask your question")
        if st.button("Submit"):
            ans = answer_question(st.session_state["vs"], query)
            st.write(ans)


if __name__ == "__main__":
    main()
