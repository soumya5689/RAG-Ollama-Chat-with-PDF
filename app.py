import streamlit as st
import logging
import os
import shutil
import pdfplumber
import ollama
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Any
from dotenv import load_dotenv
import pandas as pd
from io import BytesIO

# Load environment variables from .env file
#load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF/Excel RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=True)
def extract_model_names() -> Tuple[str, ...]:
    logger.info("Extracting model names")
    try:
        models_info = ollama.list()
        logger.info(f"Ollama list raw response: {models_info}")

        
        if hasattr(models_info, "models"):
            models_list = models_info.models
            model_names = [
                m.model
                for m in models_list
                if "embed" not in m.model.lower()
            ]
        else:
            model_names = []

        logger.info(f"Extracted chat models: {model_names}")
        return tuple(model_names)

    except Exception as e:
        logger.error(f"Error fetching model names: {e}")
        return ()


@st.cache_resource
def get_embeddings():
    logger.info("Using Ollama Embeddings with nomic-embed-text")
    # Make sure you have pulled this model with `ollama pull nomic-embed-text`
    return OllamaEmbeddings(model="nomic-embed-text")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error processing PDF: {pdf.name}. Details: {e}")
            logger.error(f"PyPDF2 error with file {pdf.name}: {e}")
    return text

def get_excel_text(excel_files):
    text = ""
    for excel_file in excel_files:
        try:
            df = pd.read_excel(excel_file, sheet_name=None)
            for sheet_name, sheet_df in df.items():
                text += f"\n--- Sheet: {sheet_name} ---\n"
                text += sheet_df.to_string(index=False)
                text += "\n"
        except Exception as e:
            st.error(f"Error processing Excel file: {excel_file.name}. Details: {e}")
            logger.error(f"Pandas error with file {excel_file.name}: {e}")
    return text

def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
    except Exception as e:
        logger.error(f"Error at get_text_chunks function: {e}")
        chunks = []
    return chunks

def get_vector_store(text_chunks):
    vector_store = None
    try:
        embeddings = get_embeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        logger.error(f"Error at get_vector_store function: {e}")
    return vector_store

@st.cache_resource
def get_llm(selected_model: str):
    logger.info(f"Getting LLM: {selected_model}")
    return ChatOllama(model=selected_model, temperature=0.1)

def process_question(question: str, vector_db: FAISS, selected_model: str) -> str:
    logger.info(f"Processing question: {question} using model: {selected_model}")
    llm = get_llm(selected_model)

    # Use simple retriever for now
    retriever = vector_db.as_retriever()
    
    # Explicitly get documents first and then format them for the prompt
    retrieved_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    logger.info(f"Retrieved {len(retrieved_docs)} docs")
    for i, doc in enumerate(retrieved_docs[:3]):
        logger.info(f"Doc {i}: {doc.page_content[:200]}")

    template = """Answer the question as detailed as possible from the provided context only.
    Do not generate a factual answer if the information is not available.
    If you do not know the answer, respond with "I don’t know the answer as not sufficient information is provided."
    Context:\n {context}\n
    Question:\n{question}\n
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({"context": context, "question": question})
    logger.info(f"LLM Raw Response: {response}")

    if not response or not response.strip():
        return "⚠️ No response generated. Check retriever/LLM."
    return response

@st.cache_data
def extract_all_pages_as_images(pdf_docs) -> List[Any]:
    pdf_pages = []
    if not pdf_docs:
        return pdf_pages
        
    for pdf_file in pdf_docs:
        logger.info(f"Extracting all pages as images from file: {pdf_file.name}")
        try:
            with pdfplumber.open(pdf_file) as pdf:
                pdf_pages.extend([page.to_image().original for page in pdf.pages])
            logger.info("PDF pages extracted as images")
        except Exception as e:
            st.error(f"Error extracting images from PDF {pdf_file.name}: {e}")
            logger.error(f"Error extracting images from {pdf_file.name}: {e}")
    return pdf_pages

def delete_vector_db() -> None:
    logger.info("Deleting vector DB")
    st.session_state.pop("pdf_pages", None)
    st.session_state.pop("excel_text", None)
    st.session_state.pop("file_upload", None)
    st.session_state.pop("vector_db", None)
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
        logger.info("FAISS index deleted")
    st.success("Collection and temporary files deleted successfully.")
    logger.info("Vector DB and related session state cleared")
    st.rerun()

def main():
    st.subheader("🧠 Ollama Chat with Documents", divider="gray", anchor=False)

    available_models = extract_model_names()
    
    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    if "file_type" not in st.session_state:
        st.session_state["file_type"] = None

    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", available_models
        )
    else:
        st.error("No compatible chat models found. Please pull a model using `ollama pull <model_name>`.")
        selected_model = None

    uploaded_files = col1.file_uploader(
        "Upload your PDF/Excel Files", 
        type=["pdf", "xlsx", "xls"],
        accept_multiple_files=True
    )

    col_buttons = col1.columns([1, 1])
    
    with col_buttons[0]:
        submit_button = st.button("Submit & Process", key="submit_process")

    with col_buttons[1]:
        delete_collection = st.button("⚠️ Delete collection", type="secondary")

    if submit_button and uploaded_files:
        st.session_state["file_type"] = uploaded_files[0].type
        with st.spinner("Processing..."):
            raw_text = ""
            if any(f.type == "application/pdf" for f in uploaded_files):
                raw_text = get_pdf_text(uploaded_files)
                st.session_state["file_type"] = "pdf"
            elif any(f.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"] for f in uploaded_files):
                raw_text = get_excel_text(uploaded_files)
                st.session_state["file_type"] = "excel"
            
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                st.session_state["vector_db"] = get_vector_store(text_chunks)
                if st.session_state["vector_db"]:
                    st.success("Done")
                
                # Check file type to decide on image extraction
                if st.session_state["file_type"] == "pdf":
                    pdf_pages = extract_all_pages_as_images(uploaded_files)
                    st.session_state["pdf_pages"] = pdf_pages
                else:
                    st.session_state["pdf_pages"] = []

    if delete_collection:
        delete_vector_db()

    with col2:
        message_container = st.container(height=500, border=True)
        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here..."):
            if st.session_state["vector_db"] is None:
                st.warning("Please upload and process a file first.")
            elif selected_model is None:
                st.warning("Please select a valid model.")
            else:
                try:
                    st.session_state["messages"].append({"role": "user", "content": prompt})
                    message_container.chat_message("user", avatar="😎").markdown(prompt)

                    with message_container.chat_message("assistant", avatar="🤖"):
                        with st.spinner(":green[processing...]"):
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                            st.session_state["messages"].append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(e, icon="⚠️")
                    logger.error(f"Error processing prompt: {e}")

    # 
    if st.session_state.get("file_type") == "pdf" and st.session_state.get("pdf_pages"):
        zoom_level = col1.slider(
            "Zoom Level", min_value=100, max_value=1000, value=700, step=50, key="zoom_slider"
        )
        with col1:
            with st.container(height=410, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)

if __name__ == "__main__":
    main()