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

# ✅ New RAG chain system
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# ------------------------------------------
# DOCUMENT PROCESSING
# ------------------------------------------
def process_input(input_type, input_data):

    # Load documents
    if input_type == "Link":
        loader = WebBaseLoader(input_data)
        docs = loader.load()
        raw_text = [doc.page_content for doc in docs]

    elif input_type == "PDF":
        pdf = PdfReader(input_data)
        text = ""
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content
        raw_text = [text]

    elif input_type == "Text":
        raw_text = [input_data]

    elif input_type == "DOCX":
        doc = Document(input_data)
        text = "\n".join(p.text for p in doc.paragraphs)
        raw_text = [text]

    elif input_type == "TXT":
        raw_text = [input_data.read().decode("utf-8")]

    # Split text
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for txt in raw_text:
        chunks.extend(splitter.split_text(txt))

    # Embeddings
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )

    # Build FAISS index
    vec = embedder.embed_query("hello")
    dim = len(vec)
    index = faiss.IndexFlatL2(dim)

    vectorstore = FAISS(
        embedding_function=embedder.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vectorstore.add_texts(chunks)
    return vectorstore


# ------------------------------------------
# QUESTION ANSWERING
# ------------------------------------------
def answer_question(vectorstore, query):

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="conversational",
        temperature=0.6,
        huggingfacehub_api_token=st.secrets["huggingface_api_key"]
    )

    chat_model = ChatHuggingFace(llm=llm)

    retriever = vectorstore.as_retriever()

    # ✅ New LangChain RAG pipeline
    doc_chain = create_stuff_documents_chain(chat_model)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    result = rag_chain.invoke({"input": query})
    return result["output_text"]


# ------------------------------------------
# STREAMLIT UI
# ------------------------------------------
def main():

    st.title("🔍 RAG Based Q&A App")

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

    if st.button("Process Document"):
        vs = process_input(input_type, input_data)
        st.session_state["vs"] = vs
        st.success("✅ Document processed successfully!")

    if "vs" in st.session_state:
        query = st.text_input("Ask your question:")
        if st.button("Submit Question"):
            answer = answer_question(st.session_state["vs"], query)
            st.write("### ✅ Answer:")
            st.write(answer)


if __name__ == "__main__":
    main()
