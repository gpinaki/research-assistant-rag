from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from operator import itemgetter
import streamlit as st
import tempfile
import os
import pandas as pd

import yaml
with open('api_credentials.yml', 'r') as file:
    api_credentials = yaml.safe_load(file)

os.environ['OPENAI_API_KEY'] = api_credentials['openai_api_key']

import locale
locale.getpreferredencoding = lambda: "UTF-8"

# Customize initial app landing page
st.set_page_config(page_title="Battery Design Bot", page_icon="🤖", layout="wide")
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
        .sidebar .sidebar-content h2 {
            color: #4CAF50;
        }
        .sidebar .sidebar-content p {
            font-size: 1.1em;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Welcome to Battery Design Assistant Bot 🤖</div>', unsafe_allow_html=True)

# Sidebar content
st.sidebar.markdown("""
    <div class="sidebar-content">
        <h2>Battery Design Assistant</h2>
        <p>Upload your PDF, Excel, or CSV files to get started. The assistant will help you answer questions based on the content of the uploaded documents.</p>
        <hr>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    """
    Takes uploaded PDFs, Excel, and CSV files, creates document chunks, computes embeddings,
    stores document chunks and embeddings in a Vector DB, and returns a retriever
    which can look up the Vector DB to return documents based on user input.
    This function is cached to improve performance.
    
    Args:
        uploaded_files (list): List of uploaded files (PDF, Excel, CSV).
    
    Returns:
        retriever: A retriever object for querying the vector database.
    """
    try:
        # Read documents
        docs = []
        temp_dir = tempfile.TemporaryDirectory()
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            
            # Handle different file types
            if file.type == "application/pdf":
                loader = PyMuPDFLoader(temp_filepath)
                docs.extend(loader.load())
            elif file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                df = pd.read_excel(temp_filepath)
                for _, row in df.iterrows():
                    docs.append({"page_content": row.to_string()})
            elif file.type == "text/csv":
                df = pd.read_csv(temp_filepath)
                for _, row in df.iterrows():
                    docs.append({"page_content": row.to_string()})
            else:
                st.warning(f"Unsupported file type: {file.type}")

        # Split into document chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        doc_chunks = text_splitter.split_documents(docs)

        # Create document embeddings and store in Vector DB
        embeddings_model = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(doc_chunks, embeddings_model)

        # Define retriever object
        retriever = vectordb.as_retriever()
        return retriever
    except Exception as e:
        st.error(f"Error configuring retriever: {e}")
        return None

class StreamHandler(BaseCallbackHandler):
    """
    Manages live updates to a Streamlit app's display by appending new text tokens
    to an existing text stream and rendering the updated text in Markdown.
    
    Args:
        container (st.delta_generator.DeltaGenerator): Streamlit container for displaying text.
        initial_text (str): Initial text to display.
    """
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Appends new token to the text stream and updates the container.
        
        Args:
            token (str): New token to append.
        """
        self.text += token
        self.container.markdown(self.text)

# Creates UI element to accept PDF, Excel, and CSV uploads
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF, Excel, or CSV files", type=["pdf", "xlsx", "xls", "csv"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload documents to continue.")
    st.stop()

# Create retriever object based on uploaded files
retriever = configure_retriever(uploaded_files)
if retriever is None:
    st.stop()

# Load a connection to GPT-4 LLM
try:
    chatgpt = ChatOpenAI(model_name='gpt-4', temperature=0.1, streaming=True)
except Exception as e:
    st.error(f"Error initializing GPT-4: {e}")
    st.stop()

# Create a prompt template for QA RAG System
qa_template = """
              You are a highly knowledgeable assistant. Use only the following pieces of context to answer the question at the end.
              If the context does not contain the answer, clearly state "I don't know based on the provided context."
              Do not attempt to fabricate an answer or provide information not included in the context.
              Keep your answer as concise and accurate as possible.

              Context:
              {context}

              Question: {question}

              Answer:
              """
qa_prompt = ChatPromptTemplate.from_template(qa_template)

def format_docs(docs):
    """
    Formats retrieved documents before sending to LLM.
    
    Args:
        docs (list): List of document objects.
    
    Returns:
        str: Formatted document content.
    """
    return "\n\n".join([d.page_content for d in docs])

# Create a QA RAG System Chain
qa_rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question")
    } | qa_prompt | chatgpt
)

# Store conversation history in Streamlit session state
streamlit_msg_history = StreamlitChatMessageHistory(key="langchain_messages")

# Shows the first message when app starts
if len(streamlit_msg_history.messages) == 0:
    streamlit_msg_history.add_ai_message("Please ask your question?")

# Render current messages from StreamlitChatMessageHistory
for msg in streamlit_msg_history.messages:
    st.chat_message(msg.type).write(msg.content)

class PostMessageHandler(BaseCallbackHandler):
    """
    Callback handler which does some post-processing on the LLM response.
    Used to post the top 3 document sources used by the LLM in RAG response.
    
    Args:
        msg (st.delta_generator.DeltaGenerator): Streamlit container for displaying sources.
    """
    def __init__(self, msg: st.delta_generator.DeltaGenerator):
        BaseCallbackHandler.__init__(self)
        self.msg = msg
        self.sources = []

    def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
        """
        Processes retrieved documents to extract metadata.
        
        Args:
            documents (list): List of retrieved document objects.
        """
        source_ids = []
        for d in documents:
            metadata = {
                "source": d.metadata["source"],
                "page": d.metadata["page"],
                "content": d.page_content[:200]
            }
            idx = (metadata["source"], metadata["page"])
            if idx not in source_ids:
                source_ids.append(idx)
                self.sources.append(metadata)

    def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
        """
        Displays the top 3 document sources used by the LLM in RAG response.
        
        Args:
            response (dict): LLM response object.
        """
        if len(self.sources):
            st.markdown("__Sources:__ " + "\n")
            st.dataframe(data=pd.DataFrame(self.sources[:3]), width=1000)

# If user inputs a new prompt, display it and show the response
if user_prompt := st.chat_input():
    st.chat_message("human").write(user_prompt)
    # This is where response from the LLM is shown
    try:
        with st.chat_message("ai"):
            # Initializing an empty data stream
            stream_handler = StreamHandler(st.empty())
            # UI element to write RAG sources after LLM response
            sources_container = st.empty()
            pm_handler = PostMessageHandler(sources_container)
            config = {"callbacks": [stream_handler, pm_handler]}
            # Get LLM response
            response = qa_rag_chain.invoke({"question": user_prompt}, config)
    except Exception as e:
        st.error(f"Error processing your request: {e}")