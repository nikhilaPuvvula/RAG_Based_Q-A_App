import streamlit as st
import faiss
import os
from io import BytesIO
from docx import Document
import numpy as np
from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint

huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

def process_input(input_type,input_data):

    loader=None
    if input_type == "Link":
        loader = WebBaseLoader(input_data)
        documents = loader.load()

    elif input_type == "PDF":
        if input_data is not None:
            pdf_reader = PdfReader(input_data)   # Directly pass the uploaded file
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # Avoid None
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
            doc = Document(input_data)   # Directly pass the uploaded file
            text = "\n".join([para.text for para in doc.paragraphs])
            documents = text
        else:
            raise ValueError("No DOCX uploaded")


    elif input_type == "TXT":
        if input_data is not None:
            text = input_data.read().decode("utf-8")   # Works directly
            documents = text
        else:
            raise ValueError("No TXT uploaded")

    
    #splitting large size documents into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    if input_type == "Link":
        texts = text_splitter.split_documents(documents)
        texts = [str(doc.page_content) for doc in texts]
    else:
        texts = text_splitter.split_text(documents)

    #getting the sentence emebeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    #creating a matching score 
    sample_embedding = np.array(hf_embeddings.embed_query("sample text"))
    dimension = sample_embedding.shape[0]
    index = faiss.IndexFlatL2(dimension)

    # Create FAISS vector store with the embedding function
    vector_store = FAISS(
        embedding_function=hf_embeddings.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Add documents to the vector store
    vector_store.add_texts(texts) 
    return vector_store

    
def answer_question(vectorstore, query):
    """Answers a question based on the provided vectorstore."""

    llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational",  
    huggingfacehub_api_token=huggingface_api_key,
    temperature=0.6
    )

    chat_llm = ChatHuggingFace(llm=llm) 
    qa = RetrievalQA.from_chain_type(llm=chat_llm,chain_type="stuff", retriever=vectorstore.as_retriever())

    answer = qa({"query": query})
    return answer


def main():

    # Custom CSS for background + selectbox theme

    # Centered title with contrasting text color
    st.markdown(
        "<h1 style='text-align: center; color: white;'>RAG Based Q&A APP</h1>",
        unsafe_allow_html=True
    )

    input_type = st.selectbox("Input Type",["Link","PDF","Text","DOCX","TXT"])
    if input_type == "Link":
        number_input = st.number_input(min_value=1, max_value=20, step=1, label = "Enter the number of Links")
        input_data = []
        for i in range(number_input):
            url = st.sidebar.text_input(f"URL {i+1}")
            input_data.append(url)

    elif input_type == "Text":
        input_data = st.text_input("Enter the text")

    elif input_type == "PDF":
        input_data = st.file_uploader("Upload a PDF file",type=["pdf"])

    elif input_type == "TXT":
        input_data = st.file_uploader("Upload a text file",type=["txt"])

    elif input_type == "DOCX":
        input_data = st.file_uploader("Upload a DOCX file",type=["docx","doc"])

    
    if st.button("Proceed"):
        # st.write(process_input(input_type, input_data))
        vectorstore = process_input(input_type, input_data)
        st.session_state["vectorstore"] = vectorstore

    if "vectorstore" in st.session_state:
        query = st.text_input("Ask your question")
        if st.button("Submit"):
            answer = answer_question(st.session_state["vectorstore"], query)
            st.write(answer["result"])
        

if __name__ == "__main__":

    main()

