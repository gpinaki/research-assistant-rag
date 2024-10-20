# Research Assistant ChatBot
# A LangChain and Streamlit-based RAG Application
# Author: Pinaki Guha
# License: MIT

"""
This application creates an intelligent research assistant chatbot using RAG (Retrieval Augmented Generation)
architecture. It processes PDF, Excel, and CSV files to provide context-aware responses using GPT-4.

Key Features:
- Document Processing: Supports PDF, Excel, and CSV files
- RAG Implementation: Uses LangChain for document processing and retrieval
- Interactive UI: Built with Streamlit for a user-friendly experience
- Real-time Streaming: Displays AI responses in real-time

Requirements:
- Python 3.8+
- OpenAI API key
- Dependencies listed in requirements.txt
"""

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from operator import itemgetter
import tempfile
import os
import pandas as pd
import yaml
import time

# Configuration and Environment Setup
def load_api_credentials():
    """Load API credentials from YAML file and set environment variables."""
    with open('api_credentials.yml', 'r') as file:
        api_credentials = yaml.safe_load(file)
    os.environ['OPENAI_API_KEY'] = api_credentials['openai_api_key']

# UI Configuration
def setup_streamlit_ui():
    """Configure Streamlit page settings and custom CSS styles."""
    st.set_page_config(
        page_title="Research Assistant ChatBot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS for better UI/UX
    st.markdown("""
        <style>
            .main-title {
                font-size: 2.5em;
                font-weight: bold;
                color: #4CAF50;
                text-align: center;
                margin-top: 20px;
            }
            .sidebar .sidebar-content {
                background-color: #f0f2f6;
                padding: 20px;
            }
            .chat-bubble {
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 10px;
            }
            .human-bubble {
                background-color: #DCF8C6;
                text-align: right;
            }
            .ai-bubble {
                background-color: #f0f2f6;
                text-align: left;
            }
        </style>
    """, unsafe_allow_html=True)

class StreamHandler(BaseCallbackHandler):
    """Handle streaming of LLM responses."""
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

class PostMessageHandler(BaseCallbackHandler):
    """Handle post-processing of messages and source attribution."""
    def __init__(self, msg: st.delta_generator.DeltaGenerator):
        BaseCallbackHandler.__init__(self)
        self.msg = msg
        self.sources = []

    def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
        """Process and store source documents for reference."""
        source_ids = []
        for d in documents:
            metadata = {
                "source": d.metadata.get("source", "N/A"),
                "page": d.metadata.get("page", "N/A"),
                "content": d.page_content[:200]
            }
            idx = (metadata["source"], metadata["page"])
            if idx not in source_ids:
                source_ids.append(idx)
                self.sources.append(metadata)

    def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
        """Display source references after LLM response."""
        if len(self.sources):
            with st.expander("__Top Sources__"):
                st.dataframe(pd.DataFrame(self.sources[:3]), width=1000)

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    """
    Configure the document retriever with uploaded files.
    
    Args:
        uploaded_files: List of uploaded file objects
        
    Returns:
        retriever: Configured retriever object for RAG
    """
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    
    with st.spinner('🔄 Processing files...'):
        for file in uploaded_files:
            # Process each file based on its type
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())

            # File processing and preview generation
            with st.expander(f"Preview: {file.name}"):
                if file.type == "application/pdf":
                    loader = PyMuPDFLoader(temp_filepath)
                    docs.extend(loader.load())
                    st.write(loader.load()[0].page_content[:1000])
                elif file.type in ["application/vnd.ms-excel", 
                                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                    df = pd.read_excel(temp_filepath)
                    st.write(df.head())
                    for _, row in df.iterrows():
                        docs.append({"page_content": row.to_string()})
                elif file.type == "text/csv":
                    df = pd.read_csv(temp_filepath)
                    st.write(df.head())
                    for _, row in df.iterrows():
                        docs.append({"page_content": row.to_string()})
                else:
                    st.warning(f"Unsupported file type: {file.type}")

        # Document processing for RAG
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        doc_chunks = text_splitter.split_documents(docs)
        
        # Create vector database
        embeddings_model = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(doc_chunks, embeddings_model)
        
        return vectordb.as_retriever()

def main():
    """Main application function."""
    # Load credentials and setup UI
    load_api_credentials()
    setup_streamlit_ui()
    
    # Sidebar and file upload
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload PDF, Excel, or CSV files",
            type=["pdf", "xlsx", "xls", "csv"],
            accept_multiple_files=True
        )

    # Process uploaded files
    if not uploaded_files:
        st.info("Please upload documents to continue.")
        st.stop()
        
    progress_bar = st.progress(0)
    progress_text = st.empty()
    retriever = configure_retriever(uploaded_files)
    progress_bar.progress(100)
    progress_text.text("Files processed successfully.")

    # Initialize ChatGPT
    try:
        chatgpt = ChatOpenAI(model_name='gpt-4', temperature=0.1, streaming=True)
    except Exception as e:
        st.error(f"Error initializing GPT-4: {e}")
        st.stop()

    # Setup RAG chain
    qa_template = """
    You are a highly knowledgeable assistant. Use the following context to answer
    the question concisely. If the context does not have the answer, respond with
    "I don't know."

    Context:
    {context}

    Question: {question}

    Answer:
    """
# This code constructs a question-answering pipeline using retrieval-augmented generation (RAG).
# It first extracts the user's question, retrieves relevant context (e.g., documents) via a retriever,
# then formats the retrieved data according to a prompt template. The processed question and context
# are passed to a GPT model (ChatGPT) to generate a response. The `|` operator is used to chain these
# operations, allowing for a clean, sequential flow from question extraction to response generation.

    qa_prompt = ChatPromptTemplate.from_template(qa_template)
    
    qa_rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question")
        } | qa_prompt | chatgpt
    )

    # Chat interface
    streamlit_msg_history = StreamlitChatMessageHistory(key="langchain_messages")
    
    if len(streamlit_msg_history.messages) == 0:
        streamlit_msg_history.add_ai_message("Please ask your question?")

    # Display chat history
    for msg in streamlit_msg_history.messages:
        bubble_class = "human-bubble" if msg.type == "human" else "ai-bubble"
        st.markdown(
            f'<div class="chat-bubble {bubble_class}">{msg.content}</div>',
            unsafe_allow_html=True
        )

    # Handle user input
    if user_prompt := st.chat_input():
        st.chat_message("human").write(user_prompt)
        with st.spinner('💡 Generating answer...'):
            try:
                stream_handler = StreamHandler(st.empty())
                sources_container = st.empty()
                pm_handler = PostMessageHandler(sources_container)
                config = {"callbacks": [stream_handler, pm_handler]}
                response = qa_rag_chain.invoke({"question": user_prompt}, config)
            except Exception as e:
                st.error(f"Error processing your request: {e}")

if __name__ == "__main__":
    main()