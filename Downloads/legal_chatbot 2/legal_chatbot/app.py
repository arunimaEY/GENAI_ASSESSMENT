import streamlit as st
import google.generativeai as genai
import pandas as pd
import io
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from streamlit_chat import message
from config import GEMINI_API_KEY
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

# ✅ Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Set Page Config
st.set_page_config(page_title="Legal AI Chatbot", page_icon="⚖️", layout="wide")

# ✅ Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# ✅ Load Custom CSS
def load_custom_css():
    with open("static/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()

# ✅ Extract Text from Uploaded File
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1]
    text = ""

    if file_type == "pdf":
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

    elif file_type == "docx":
        doc = DocxDocument(io.BytesIO(uploaded_file.read()))
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif file_type == "txt":
        text = uploaded_file.read().decode("utf-8")

    elif file_type in ["xls", "xlsx"]:
        df = pd.read_excel(uploaded_file)
        text = df.to_string(index=False)

    return text

# ✅ Chunk Text for RAG Processing
def chunk_text(text, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# ✅ Store Chunks in Vector Database
def store_in_vector_db(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # ✅ Local Embeddings
    vectorstore = FAISS.from_texts(chunks, embedding=embedding_model)
    return vectorstore

# ✅ Retrieve Relevant Chunks
def retrieve_relevant_chunks(vectorstore, query, top_k=3):
    results = vectorstore.similarity_search(query, k=top_k)
    return "\n\n".join([doc.page_content for doc in results])

# ✅ Generate AI Response with RAG
def get_rag_based_response(query, vectorstore):
    relevant_text = retrieve_relevant_chunks(vectorstore, query)
    prompt = f"Using the following document context, answer the question: {query}\n\n{relevant_text}"
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

# ✅ Sidebar: Chat History + Controls
with st.sidebar:
    st.markdown("<div class='Chatbot'><span>⚖️</span> <span class='LegalChatbot'>Legal AI Chatbot</span></div>", unsafe_allow_html=True)

# Custom styled "New Chat" button
    st.markdown('<div class="new-chat-button">', unsafe_allow_html=True)
    if st.button("➕ New Chat"):
       st.session_state.chat_history = []  # ✅ Clear chat history
       st.session_state.current_doc = None  # ✅ Reset document
       st.session_state.uploaded_file = None  # ✅ Reset uploaded file
       st.experimental_set_query_params()  # ✅ Force refresh to reset file uploader
       st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Chat History")
    if st.session_state.chat_history:
        for i, (user_q, _) in enumerate(st.session_state.chat_history):
            st.button(f"🔹 {user_q[:30]}...", key=f"history_{i}")

import base64
import streamlit as st

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_base64 = get_base64_image("static/images/logo.png")

# ✅ Corrected Markdown for Main Title with Logo
st.markdown(
    f"<h1><img src='data:image/png;base64,{image_base64}' width='150' style='vertical-align: middle;' /> Legal AI Chatbot</h1>",
    unsafe_allow_html=True
)


# ✅ File Upload
st.markdown("### 📂 Upload a Legal Document")



# st.session_state.uploaded_file = st.file_uploader("Upload (PDF, DOCX, TXT, Excel)", type=["pdf", "docx", "txt", "xls", "xlsx"])
uploaded_file = st.file_uploader("Upload (PDF, DOCX, TXT, Excel)", type=["pdf", "docx", "txt", "xls", "xlsx"])
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file  # ✅ Store the uploaded file

if st.session_state.uploaded_file is not None:
    if st.session_state.current_doc != st.session_state.uploaded_file.name:
        st.session_state.current_doc = st.session_state.uploaded_file.name
        st.session_state.chat_history = []  # ✅ Reset chat when document changes

    with st.spinner("🔄 Processing document... Please wait"):
        document_text = extract_text_from_file(st.session_state.uploaded_file)
        chunks = chunk_text(document_text)
        vector_db = store_in_vector_db(chunks)

    st.success(f"✅ {st.session_state.uploaded_file.name} uploaded & indexed!")

    with st.expander("📄 View Extracted Document Text"):
        st.text_area("Extracted Content:", document_text, height=250)

    # ✅ Chat Interface with Multi-Turn Support
    st.markdown("### 💬 Ask a Legal Question")

    # ✅ Display Chat History
    for i, (user_q, ai_a) in enumerate(st.session_state.chat_history):
        message(user_q, is_user=True, key=f"user_{i}")
        message(ai_a, is_user=False, key=f"ai_{i}")

    # ✅ Input for New Questions
    user_question = st.text_input("🔍 Type your legal query:", key=f"input_{len(st.session_state.chat_history)}")

    if st.button("🔎 Get Answer"):
        if user_question:
            with st.spinner("⚖️ AI is retrieving relevant legal text..."):
                response = get_rag_based_response(user_question, vector_db)
                st.session_state.chat_history.append((user_question, response))  # ✅ Store conversation history

            st.rerun()  # ✅ Refresh UI for new question
        else:
            st.warning("⚠️ Please enter a question before clicking 'Get Answer'.")

# ✅ Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>⚖️ Built with ❤️ by KAADS</div>", unsafe_allow_html=True)
