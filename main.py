import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import os
import tempfile

# Define prompts
contextualize_q_system_prompt = """Given a chat history and the latest user question, if the question is out of context or history, respond with 'Out of context'. Otherwise, reformulate the question to make it standalone."""
contextualize_q_prompt = ChatPromptTemplate.from_messages([ 
    ("system", contextualize_q_system_prompt), 
    MessagesPlaceholder("chat_history"), 
    ("human", "{input}")
])

system_prompt = (
    "You are a helpful assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If the retrieved context does not provide enough information or is irrelevant to the question, respond with 'No context found'. "
    "Do not attempt to answer if the context does not fully support the answer. "
    "Limit the answer to a maximum of ten sentences and ensure that it is clear, concise, and directly relevant to the retrieved context from the provided PDF. "
    "If no relevant context is found, respond with 'No context found.'"
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Function to handle PDF upload and processing
def handle_pdf_upload(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    return documents

# Initialize session state
if 'docs' not in st.session_state:
    st.session_state.docs = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'chatHistory' not in st.session_state:
    st.session_state.chatHistory = []


OPEN_API = st.secrets["OPEN_KEY"]
GOOGLE_API = st.secrets["GOOGLE_KEY"]
LANGSMITH_API = st.secrets["LANGSMITH_KEY"]

# Set API keys
os.environ["OPENAI_API_KEY"] = OPEN_API
os.environ["GOOGLE_API_KEY"] = GOOGLE_API
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langSmith"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API

# Initialize language models
if 'gemini_llm' not in st.session_state:
    st.session_state.gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
if 'gpt_llm' not in st.session_state:
    st.session_state.gpt_llm = ChatOpenAI(model="gpt-4")


st.markdown("""
    <style>
        .big-font {
            font-size: 50px;
            font-weight: bold;
            color: #1E90FF;
        }
        .desc-font {
            font-size: 20px;
            color: #4B0082;
        }
        .highlight {
            background-color: #FFFF00;
            padding: 0 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Description with enhanced styling
st.markdown('<p class="big-font">GenAnswer</p>', unsafe_allow_html=True)
st.markdown('<p class="desc-font">✨ Your Personal AI Assistant with <span class="highlight">Retrieval-Augmented Generation</span></p>', unsafe_allow_html=True)

# Model selection sidebar
selected_model = st.sidebar.radio("Choose a model:", options=["Gemini", "GPT"])

# Move the file uploader to the bottom of the sidebar
with st.sidebar:
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

# Handle PDF upload
if uploaded_file is not None:
    # Reset relevant session states but retain chat history
    st.session_state.docs = handle_pdf_upload(uploaded_file)
    st.session_state.vectorstore = None
    st.session_state.rag_chain = None

    # Split documents and create vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(st.session_state.docs)
    vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    st.session_state.vectorstore = vectorstore

# Update RAG chain if the model or documents are updated
if st.session_state.vectorstore is not None and selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    llm = st.session_state.gemini_llm if selected_model == 'Gemini' else st.session_state.gpt_llm
    retriever = st.session_state.vectorstore.as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    st.session_state.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Ensure RAG chain is initialized after both the model and PDF are uploaded
if st.session_state.vectorstore is not None and st.session_state.selected_model is not None:
    if st.session_state.rag_chain is None:
        llm = st.session_state.gemini_llm if st.session_state.selected_model == 'Gemini' else st.session_state.gpt_llm
        retriever = st.session_state.vectorstore.as_retriever()
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        st.session_state.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Display previous chat history
for message in st.session_state.chatHistory:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Handle user input and interaction
user_input = st.chat_input(placeholder="Type your question here")

if user_input:
    # Add user input to the chat history
    st.session_state.chatHistory.append(HumanMessage(content=user_input))
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response using RAG chain
    if st.session_state.rag_chain is not None:
        with st.spinner("Generating response..."):
            results = st.session_state.rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chatHistory})
            answer = results['answer']
            
            # Add bot response to the chat history
            st.session_state.chatHistory.append(AIMessage(content=answer))
            
            # Display bot response
            with st.chat_message("assistant"):
                st.write(answer)
    else:
        st.warning("RAG chain not initialized. Please upload a PDF and select a model.")