Ollama Chat with Documents
-------------------------

This is a Streamlit-based application that creates a conversational AI assistant capable of answering questions 
based on the content of uploaded PDF and Excel files. The application uses a Retrieval-Augmented Generation(RAG) 
system powered by local Ollama models and LangChain.


Features
--------
Multi-document Support: Process and chat with both PDF and Excel files.
Local LLMs: Utilizes Ollama to run large language models (LLMs) and embeddings locally, ensuring data privacy and offline functionality.
Vector Storage: Employs FAISS to create a vector store from the document content, enabling efficient retrieval of relevant information.
Dynamic Model Selection: Allows users to select from a list of compatible chat models available on their local Ollama server.
Document Visualization: Renders PDF pages as images within the UI for easy reference.
Clean-up Functionality: Includes an option to delete the current vector collection and temporary files.

Prerequisites
--------------
Before running the application, ensure you have the following installed:
Python 3.8+
Ollama: A running Ollama server with at least one compatible chat model and the nomic-embed-text model pulled.
To pull the required models, run the following commands in your terminal:
Bash
ollama pull nomic-embed-text
ollama pull <your-chosen-model>
Replace <your-chosen-model> with a model name like llama3, mistral, or gemma.


requirements.txt file that lists the necessary Python libraries:
-------------------------------------------------------------
streamlit
pdfplumber
PyPDF2
ollama
langchain
langchain-ollama
faiss-cpu
python-dotenv
pandas
openpyxl
To install these dependencies, you can run the following command in your terminal:
pip install -r requirements.txt


Start the Ollama Server: Make sure your Ollama server is running.
Run the Streamlit App: Open your terminal in the project directory and run the following command:
Bash
streamlit run your_app_file_name.py
(Note: Replace your_app_file_name.py with the name you saved the code file as.)


Interact with the UI:
---------------------
Upload one or more PDF or Excel files.
Select a chat model from the dropdown list.
Click "Submit & Process" to create the document vector store.
Once processing is complete, use the chat input at the bottom to ask questions about your documents.
Use the "⚠️ Delete collection" button to clear the loaded files and start fresh.
