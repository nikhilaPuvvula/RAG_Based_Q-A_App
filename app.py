import streamlit as st
import numpy as np
import faiss
from docx import Document
from PyPDF2 import PdfReader

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


# -----------------------------
# DOCUMENT PROCESSING
# -----------------------------
def process_input(input_type, input_data):

    if input_type == "Link":
        docs = WebBaseLoader(input_data).load()
        raw_text = [d.page_content for d in docs]

    elif input_type == "PDF":
        pdf = PdfReader(input_data)
        text = "".join(page.extract_text() or "" for page in pdf.pages)
        raw_text = [text]

    elif input_type == "Text":
        raw_text = [input_data]

    elif input_type == "DOCX":
        doc = Document(input_data)
        raw_text = ["\n".join(p.text for p in doc.paragraphs)]

    elif input_type == "TXT":
        raw_text = [input_data.read().decode("utf-8")]

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for txt in raw_text:
        chunks.extend(splitter.split_text(txt))

    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
    )

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


# -----------------------------
# RAG ANSWERING
# -----------------------------
def answer_question(vectorstore, query):

    retriever = vectorstore.as_retriever()

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="conversational",
        huggingfacehub_api_token=st.secrets["huggingface_api_key"],
    )

    chat_model = ChatHuggingFace(llm=llm)

    prompt = PromptTemplate.from_template("""
Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:
""")

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | chat_model
    )

    return rag_chain.invoke(query)


# -----------------------------
# STREAMLIT UI
# -----------------------------
def main():

    st.title("RAG Based Q&A App (LangChain 0.4.x)")

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
        st.session_state["vs"] = process_input(input_type, input_data)
        st.success("✅ Document processed!")

    if "vs" in st.session_state:
        query = st.text_input("Ask your question:")
        if st.button("Submit"):
            answer = answer_question(st.session_state["vs"], query)
            st.write(answer)


if __name__ == "__main__":
    main()
