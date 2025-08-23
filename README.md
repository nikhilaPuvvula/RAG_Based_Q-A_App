# RAG Based Q&A App

A **Streamlit-based Retrieval-Augmented Generation (RAG) Q&A application** that allows users to upload documents or provide links, processes them into embeddings, and answers natural language queries using **Meta-LLaMA-3** via HuggingFace.

---

##  Features
- **Multiple Input Sources**: Supports **PDF, DOCX, TXT, plain text, and web links**.  
- **Semantic Search with FAISS**: Chunks documents and stores embeddings for efficient retrieval.  
- **LLM-Powered Responses**: Uses **Meta-LLaMA-3-8B-Instruct** through HuggingFace Inference API.  
- **User-Friendly Interface**: Built with Streamlit for easy interaction.  
- **Custom RAG Pipeline**: Combines retrieval with LLM for accurate answers.

---

##  Tech Stack
- [Streamlit](https://streamlit.io/) – UI framework
- [LangChain](https://www.langchain.com/) – RetrievalQA pipeline
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) – Text embeddings
- [FAISS](https://faiss.ai/) – Vector search
- [PyPDF2](https://pypi.org/project/PyPDF2/) & [python-docx](https://pypi.org/project/python-docx/) – File parsing
- [Meta-LLaMA-3](https://huggingface.co/meta-llama) – LLM for answering queries  

---
## Pipeline of this Project

<img width="998" height="291" alt="image" src="https://github.com/user-attachments/assets/add7d64c-2e6d-4adb-918c-22ad396f1c1f" />
---


##  Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/rag-qa-app.git
   cd rag-qa-app
2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your HuggingFace API key:
   Create a file secret_api_keys.py in the project root:
   ```python
   huggingface_api_key = "your_huggingface_api_key_here"
   ```
   
## Usage

Run the Streamlit app:

  ```bash
  streamlit run app.py
  ```

1. Select an input type (Link, PDF, DOCX, TXT, or Text).

2. Upload or provide the content.

3. Click Proceed to process.

4. Ask your question in natural language.

5. Get precise, LLM-powered answers!

Example Use Cases

1. Question answering from research papers

2. Summarizing and querying policy/contract documents

3. Extracting insights from multiple web articles

4. Semantic Q&A on personal notes or reports

----


