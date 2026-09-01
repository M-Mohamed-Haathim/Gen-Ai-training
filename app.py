import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

PROVIDER_MODELS = {
    "Groq": "openai/gpt-oss-120b",
    "OpenAI": "gpt-4o-mini",
    "Gemini": "gemini-2.0-flash",
}

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG ChatBot",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Modern UI styling — Gemini (soft gradients, rounded cards) x ChatGPT (dark,
# spacious chat layout)
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    @keyframes ambientGlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 8px rgba(212, 175, 55, 0.5), 0 0 0 rgba(212, 175, 55, 0); }
        50% { box-shadow: 0 0 18px rgba(212, 175, 55, 0.85), 0 0 6px rgba(212, 175, 55, 0.4); }
    }
    @keyframes borderRotate {
        100% { --angle: 360deg; }
    }
    @property --angle {
        syntax: '<angle>';
        initial-value: 0deg;
        inherits: false;
    }

    .stApp {
        background:
            radial-gradient(circle at 20% -10%, rgba(212,175,55,0.10) 0%, transparent 40%),
            radial-gradient(circle at 85% 110%, rgba(212,175,55,0.06) 0%, transparent 45%),
            linear-gradient(160deg, #060606 0%, #0a0908 50%, #000000 100%);
        background-size: 200% 200%;
        animation: ambientGlow 18s ease infinite;
    }

    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stAppViewContainer"] { background: transparent; }
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"] > div,
    .stChatFloatingInputContainer,
    [data-testid="stChatInputContainer"] {
        background: linear-gradient(180deg, rgba(0,0,0,0) 0%, #000000 45%) !important;
    }
    main [data-testid="stVerticalBlockBorderWrapper"] { background: transparent; }

    /* Sidebar — black glass with gold edge */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0b09 0%, #060605 100%);
        border-right: 1px solid #2b2416;
        box-shadow: 4px 0 24px rgba(0,0,0,0.6);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
    }

    /* Shimmering gold title, Cinzel for a touch of luxe */
    .app-title {
        font-family: 'Cinzel', serif;
        font-weight: 700;
        font-size: 1.55rem;
        background: linear-gradient(90deg, #b8860b 0%, #ffd700 25%, #fff3b0 50%, #ffd700 75%, #b8860b 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 5s linear infinite;
        margin-bottom: 0;
        letter-spacing: 0.02em;
    }
    .app-subtitle {
        color: #8a7f66;
        font-size: 0.85rem;
        margin-top: 4px;
        margin-bottom: 1.4rem;
    }
    .main-header {
        padding: 0.5rem 0 1.2rem 0;
        border-bottom: 1px solid #2b2416;
        margin-bottom: 1rem;
    }

    /* Chat bubbles — 3D-ish elevation with gold-tinted shadow, fade-in on entry */
    .stChatMessage {
        border-radius: 18px;
        padding: 12px 16px;
        margin-bottom: 12px;
        animation: fadeInUp 0.35s ease;
        transform-style: preserve-3d;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stChatMessage:hover {
        transform: translateY(-2px);
    }
    div[data-testid="stChatMessageContent"] {
        font-size: 0.96rem;
        line-height: 1.65;
        color: #f2ead9;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(145deg, #171410, #0d0c0a);
        border: 1px solid #33291a;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 6px 20px rgba(0,0,0,0.5);
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        background: linear-gradient(145deg, #14110a, #0a0906);
        border: 1px solid #3a2f18;
        box-shadow: inset 0 1px 0 rgba(255,215,0,0.04), 0 8px 24px rgba(212,175,55,0.08);
    }
    div[data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #b8860b, #ffd700) !important;
        animation: pulseGlow 3s ease-in-out infinite;
    }
    div[data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #3a3427, #1a1712) !important;
        border: 1px solid #4a3f24 !important;
    }

    /* Pill chat input with a slowly rotating gold gradient ring */
    div[data-testid="stChatInput"] {
        border-radius: 28px !important;
        background: #0c0b09 !important;
        position: relative;
        padding: 2px;
        border: 1px solid transparent;
        background-image:
            linear-gradient(#0c0b09, #0c0b09),
            conic-gradient(from var(--angle), #4a3f24, #ffd700, #4a3f24, #ffd700, #4a3f24);
        background-origin: border-box;
        background-clip: padding-box, border-box;
        animation: borderRotate 6s linear infinite;
    }
    div[data-testid="stChatInput"] textarea {
        border-radius: 26px !important;
        background-color: transparent !important;
        border: none !important;
        color: #f2ead9 !important;
    }
    div[data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #b8860b, #ffd700) !important;
        border-radius: 50% !important;
        border: none !important;
        box-shadow: 0 0 10px rgba(255,215,0,0.4);
    }

    /* Buttons — gold-edged, 3D press effect */
    .stButton > button {
        border-radius: 14px;
        border: 1px solid #3a2f18;
        background: linear-gradient(145deg, #16130d, #0c0a07);
        color: #e8dcc0;
        box-shadow: 0 3px 0 #050403, 0 6px 14px rgba(0,0,0,0.4);
        transition: all 0.12s ease;
    }
    .stButton > button:hover {
        border-color: #ffd700;
        color: #ffd700;
        box-shadow: 0 3px 0 #050403, 0 0 14px rgba(255,215,0,0.25);
    }
    .stButton > button:active {
        transform: translateY(3px);
        box-shadow: 0 0 0 #050403;
    }

    /* Inputs and selects */
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] > div {
        border-radius: 14px !important;
        background-color: #0c0b09 !important;
        border: 1px solid #3a2f18 !important;
        color: #f2ead9 !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #ffd700 !important;
        box-shadow: 0 0 0 2px rgba(255,215,0,0.2) !important;
    }
    div[data-testid="stTextInput"] > div,
    div[data-testid="stTextInputRootElement"] {
        background-color: #0c0b09 !important;
        border-radius: 14px !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #0c0b09 !important;
        border-color: #3a2f18 !important;
    }

    /* File uploader — dashed gold drop zone */
    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 16px;
        background-color: #0c0b09 !important;
        border: 1.5px dashed #4a3f24;
        transition: border-color 0.2s ease;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #ffd700;
    }
    div[data-testid="stFileUploaderDropzone"] section {
        background-color: transparent !important;
    }
    div[data-testid="stFileUploaderDropzone"] button {
        border-radius: 12px !important;
        background: linear-gradient(135deg, #b8860b, #ffd700) !important;
        color: #0a0906 !important;
        font-weight: 600;
        border: none !important;
        box-shadow: 0 0 10px rgba(255,215,0,0.3);
    }
    div[data-testid="stFileUploaderDropzone"] small,
    div[data-testid="stFileUploaderDropzone"] span {
        color: #8a7f66 !important;
    }

    /* Source pills — thin gold outline chips */
    .source-pill {
        display: inline-block;
        background: rgba(255,215,0,0.06);
        color: #ffd700;
        border: 1px solid #4a3f24;
        border-radius: 999px;
        padding: 3px 12px;
        font-size: 0.72rem;
        margin: 3px 5px 3px 0;
    }

    div[data-testid="stExpander"] {
        border-radius: 14px;
        border: 1px solid #2b2416;
        background-color: #0c0b09;
    }

    /* Uploaded-file rows */
    div[data-testid="stFileUploaderFile"] {
        background-color: #0e0c09;
        border: 1px solid #2b2416;
        border-radius: 10px;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_llm" not in st.session_state:
    st.session_state.current_llm = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm(provider: str, api_key: str):
    model = PROVIDER_MODELS[provider]
    if provider == "Groq":
        return ChatGroq(model=model, temperature=0.2, groq_api_key=api_key)
    elif provider == "OpenAI":
        return ChatOpenAI(model=model, temperature=0.2, api_key=api_key)
    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(model=model, temperature=0.2, google_api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def load_document(uploaded_file) -> list:
    """Save an uploaded file to a temp path and load it with the right loader."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix == ".csv":
        loader = CSVLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)

    docs = loader.load()
    # Keep the original filename in metadata instead of the temp path
    for d in docs:
        d.metadata["source"] = uploaded_file.name
    return docs


RAG_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context below. If the context
doesn't contain the answer, say you don't know based on the uploaded documents.

Context:
{context}

Question: {question}

Answer:"""
)


def answer_question(query: str, retriever, llm) -> dict:
    """Retrieve relevant chunks and generate an answer, replacing the old
    RetrievalQA chain (removed from core langchain as of v1.0)."""
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    chain = RAG_PROMPT | llm
    response = chain.invoke({"context": context, "question": query})
    return {"result": response.content, "source_documents": docs}


def build_or_update_index(uploaded_files, embeddings):
    all_new_docs = []
    for f in uploaded_files:
        if f.name in st.session_state.indexed_files:
            continue
        all_new_docs.extend(load_document(f))
        st.session_state.indexed_files.append(f.name)

    if not all_new_docs:
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(all_new_docs)

    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        st.session_state.vectorstore.add_documents(chunks)


# --------------------------------------------------------------------------
# Sidebar — API key, file upload, controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="app-title">◆ RAG ChatBot</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Chat with your documents</p>', unsafe_allow_html=True)

    provider = st.selectbox("AI Provider", list(PROVIDER_MODELS.keys()))

    api_key = st.text_input(
        f"{provider} API Key",
        type="password",
        help="Your key is used only for this session and is never stored or logged.",
    )

    with st.expander("Where do I get a key?"):
        st.markdown(
            "- **Groq**: console.groq.com/keys (free, fast)\n"
            "- **OpenAI**: platform.openai.com/api-keys\n"
            "- **Gemini**: aistudio.google.com/apikey (free tier available)"
        )

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files and api_key:
        with st.spinner("Indexing documents..."):
            embeddings = get_embeddings()
            build_or_update_index(uploaded_files, embeddings)
            st.session_state.current_llm = get_llm(provider, api_key)

    # If docs are already indexed but provider/key changed, rebuild just the LLM
    elif api_key and st.session_state.vectorstore is not None:
        current = (provider, api_key)
        if st.session_state.get("_last_llm_config") != current:
            st.session_state.current_llm = get_llm(provider, api_key)
    st.session_state["_last_llm_config"] = (provider, api_key)

    if st.session_state.indexed_files:
        st.caption("Indexed files:")
        for name in st.session_state.indexed_files:
            st.markdown(f'<span class="source-pill">{name}</span>', unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("♻️ Reset everything (chat + documents)", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.current_llm = None
        st.session_state.indexed_files = []
        st.rerun()

# --------------------------------------------------------------------------
# Main chat area
# --------------------------------------------------------------------------
st.markdown(
    '<div class="main-header">'
    '<p class="app-title">Chat</p>'
    '<p class="app-subtitle">Ask anything about your uploaded documents</p>'
    '</div>',
    unsafe_allow_html=True,
)

if not api_key:
    st.info("👋 Enter your API key in the sidebar to get started.")
elif not st.session_state.indexed_files:
    st.info("📄 Upload a document (PDF, TXT, or CSV) in the sidebar to start chatting.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Send a message...")

if prompt:
    if not api_key:
        st.warning("Please enter your API key in the sidebar first.")
    elif st.session_state.vectorstore is None or st.session_state.current_llm is None:
        st.warning("Please upload a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                    result = answer_question(prompt, retriever, st.session_state.current_llm)
                    answer = result["result"]
                    sources = result.get("source_documents", [])
                except Exception as e:
                    answer = f"Something went wrong answering that: {e}"
                    sources = []

                if sources:
                    seen = set()
                    pills = ""
                    for doc in sources:
                        label = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page")
                        tag = f"{label}" + (f" (p.{page + 1})" if isinstance(page, int) else "")
                        if tag not in seen:
                            seen.add(tag)
                            pills += f'<span class="source-pill">{tag}</span>'
                    answer_display = answer + f"\n\n{pills}"
                else:
                    answer_display = answer

                st.markdown(answer_display, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer_display})
