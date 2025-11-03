import streamlit as st
from dotenv import load_dotenv
import os
import time
from pathlib import Path
import datetime
from typing import Optional, Tuple
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
import torch
from langchain_community.document_loaders import PyPDFDirectoryLoader
# from collections import deque

# Constants
VECTOR_STORE_PATH = Path("./vector_store.pkl")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
USE_GPU = torch.cuda.is_available()
MAX_HISTORY = 5

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(page_title="AI Legal Assistant", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of messages; we'll keep at most MAX_HISTORY entries
if "processing" not in st.session_state:
    st.session_state.processing = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# Lazy cached embeddings with proper GPU support
@st.cache_resource
def get_embeddings():
    if not st.session_state.embeddings:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if USE_GPU else 'cpu'}
        )
    return st.session_state.embeddings

# Set up OpenAI API key
os.environ['GROQ_API_KEY'] = 'gsk_rEtgnc9pmDAsueR7TxfcWGdybFYNZqezBcA5jmeatbXAhnm4TRx'
# Initialize Wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Optimized vector store initialization with persistence
@st.cache_resource
def initialize_vector_store():
    """Load or create FAISS vectorstore with persistent storage."""
    if st.session_state.vector_store:
        return st.session_state.vector_store

    embeddings = get_embeddings()

    # Try to load from disk first
    if VECTOR_STORE_PATH.exists():
        try:
            with open(VECTOR_STORE_PATH, 'rb') as f:
                vectorstore = pickle.load(f)
                st.session_state.vector_store = vectorstore
                return create_retriever_tools(vectorstore)
        except Exception as e:
            st.warning(f"Could not load vector store from disk: {e}")

    # If loading fails, create new vectorstore
    try:
        documents = []
        
        # Try loading from PDF directory first
        if Path("./data").exists():
            loader = PyPDFDirectoryLoader("./data")
            documents.extend(loader.load()[:50])  # Limit initial load
        
        # Try loading individual PDF if exists
        if Path("practice.pdf").exists():
            loader = PyPDFLoader("practice.pdf")
            documents.extend(loader.load())
        
        # Try web loader as fallback
        if not documents:
            loader = WebBaseLoader("https://lawmin.gov.in/")
            documents.extend(loader.load()[:20])  # Limit pages
        
        if not documents:
            st.error("No documents found to index")
            return None, None

        # Process documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Save to disk
        with open(VECTOR_STORE_PATH, 'wb') as f:
            pickle.dump(vectorstore, f)
        
        st.session_state.vector_store = vectorstore
        return create_retriever_tools(vectorstore)

    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None, None

def create_retriever_tools(vectorstore):
    """Create retriever tools from a vectorstore."""
    if not vectorstore:
        return None, None
        
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Use MMR for better diversity
        search_kwargs={"k": 3}
    )
    
    retriever_tool = create_retriever_tool(
        retriever,
        "law_search",
        "Search for information about Indian laws and Supreme Court judgments."
    )
    
    # Create PDF-specific retriever if PDF exists
    pretriever_tool = None
    if Path("practice.pdf").exists():
        pdf_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "filter": {"source": "practice.pdf"}}
        )
        pretriever_tool = create_retriever_tool(
            pdf_retriever,
            "law_search_by_pdf",
            "Search Supreme Court procedures and sections from PDF."
        )
    
    return retriever_tool, pretriever_tool


def get_vector_tools():
    return initialize_vector_store()

# Initialize Arxiv tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

@st.cache_resource
def get_llm():
    # Lazily initialize and cache the LLM to avoid blocking startup
    return ChatGroq(model="qwen/qwen3-32b", temperature=0.3, request_timeout=30)

# Define custom system prompt
custom_system_prompt = """You are an AI legal assistant that helps users understand legal matters and provides information about laws, court judgments, and legal procedures.

Follow these structured response guidelines:

# Response Format
1. **Case Summary / Query Overview**
   - Brief description of the situation
   - Key legal issues identified

2. **Immediate Actions Required**
   - Step-by-step actionable items
   - Priority level for each action
   - Relevant authorities to contact

3. **Applicable Laws & Regulations**
   - Relevant acts and sections
   - Important supreme court judgments
   - State-specific regulations (if applicable)

4. **Legal Procedures**
   - Filing requirements
   - Timeline expectations
   - Required documentation

5. **Professional Guidance**
   - Type of legal professional to consult
   - Additional resources
   - Legal aid options if available

6. **Important Notes**
   - Limitations and disclaimers
   - Time-sensitive considerations
   - Risk factors

Rules:
- Always provide citations for legal references
- Use simple, non-technical language when possible
- Include relevant case laws and precedents
- Clearly mark urgent actions
- Always include a disclaimer"""

# Create custom prompt template
custom_legal_prompt = ChatPromptTemplate.from_messages([
    ("system", custom_system_prompt),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

@st.cache_resource
def get_agent_executor():
    # Build agent lazily: create vector tools and llm when first needed
    llm = get_llm()
    retriever_tool, pretriever_tool = get_vector_tools()
    tools = [t for t in (pretriever_tool, retriever_tool, wiki, arxiv) if t]
    agent = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=custom_legal_prompt.partial(chat_history=[], tools_instructions="")
    )
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit UI (WhatsApp-like chat)
st.title("AI Legal Assistant")

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {role, content, ts}
if "processing" not in st.session_state:
    st.session_state.processing = False

# Persona selector in sidebar
persona = st.sidebar.selectbox("Reply style / persona", [
    "Layperson (simple, non-technical)",
    "Legal Professional (detailed, citations)",
    "Concise (short answers)",
    "Verbose (explain step-by-step)"
])

# Chat CSS
st.markdown("""
<style>
.chat-row { display:flex; margin:6px 0; }
.bubble { padding:10px 14px; border-radius:16px; max-width:75%; color:#222; }
.user { background:#DCF8C6; margin-left:auto; }
.assistant { background:#F1F0F0; margin-right:auto; }
.meta { font-size:10px; color:#444; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# Show chat (oldest -> newest) so newest is at bottom like WhatsApp
for m in st.session_state.chat_history:
    role = m.get("role")
    content = m.get("content")
    ts = m.get("ts")
    timestr = datetime.datetime.fromtimestamp(ts).strftime("%H:%M") if ts else ""
    if role == "user":
        st.markdown(f"<div class='chat-row'><div class='bubble user'>" + content + f"<div class='meta'>{timestr}</div></div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-row'><div class='bubble assistant'>" + content + f"<div class='meta'>{timestr}</div></div></div>", unsafe_allow_html=True)

# Handle input clearing on submit
def clear_input():
    st.session_state.user_input = ""

# Input area
col1, col2 = st.columns([9,1])
with col1:
    query = st.text_input("", 
                         key="user_input", 
                         placeholder='Type your question here...',
                         on_change=clear_input if st.session_state.get('clear_on_submit', False) else None)
    # Reset the clear flag
    st.session_state.clear_on_submit = False
    
with col2:
    send = st.button('Send', disabled=st.session_state.processing)

if send and query.strip():
    st.session_state.processing = True
    ts = time.time()
    current_query = query  # Store current query before clearing
    
    # Set flag to clear on next render
    st.session_state.clear_on_submit = True
    
    # Optimistic append (keep history size bounded)
    st.session_state.chat_history.append({"role": "user", "content": current_query, "ts": ts})
    if len(st.session_state.chat_history) > MAX_HISTORY:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]
    
    try:
        with st.spinner('Thinking...'):
            agent_executor = get_agent_executor()
            persona_note = f"Respond as: {persona}."
            prompt_input = persona_note + "\n" + current_query
            # Pass chat_history as a plain list (agent expects a list of messages)
            res = agent_executor.invoke({"input": prompt_input, "chat_history": list(st.session_state.chat_history)})
            
            # Try to extract text from result
            answer = ''
            if isinstance(res, dict):
                answer = res.get('output') or res.get('text') or res.get('answer') or ''
            else:
                answer = str(res)
            
            if not answer:
                # Fallback to direct llm call
                llm = get_llm()
                try:
                    gen = llm.generate({"input": prompt_input})
                    # Extract best text
                    if isinstance(gen, dict):
                        answer = gen.get('output') or gen.get('text') or str(gen)
                    else:
                        answer = str(gen)
                except Exception:
                    answer = '(No answer)'
            
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "ts": time.time()})
            if len(st.session_state.chat_history) > MAX_HISTORY:
                st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY:]
            st.rerun()  # Rerun to update UI with new message
            
    except Exception as e:
        st.error(f'Error: {e}')
    finally:
        st.session_state.processing = False
