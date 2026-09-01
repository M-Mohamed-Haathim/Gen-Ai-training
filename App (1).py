import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

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
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Modern dark UI styling (ChatGPT-inspired)
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
    }
    section[data-testid="stSidebar"] {
        background-color: #171a21;
        border-right: 1px solid #2a2d34;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 4px 8px;
    }
    div[data-testid="stChatMessageContent"] {
        font-size: 0.95rem;
        line-height: 1.55;
    }
    .app-title {
        font-weight: 700;
        font-size: 1.4rem;
        background: linear-gradient(90deg, #10a37f, #1a7f64);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .app-subtitle {
        color: #9ba1a6;
        font-size: 0.85rem;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }
    .source-pill {
        display: inline-block;
        background-color: #1f2937;
        color: #9ca3af;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px 4px 2px 0;
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
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
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
    st.markdown('<p class="app-title">🤖 RAG ChatBot</p>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Powered by Groq + FAISS</p>', unsafe_allow_html=True)

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
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=get_llm(provider, api_key),
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=True,
            )

    # If docs are already indexed but provider/key changed, rebuild just the chain
    elif api_key and st.session_state.vectorstore is not None:
        current = (provider, api_key)
        if st.session_state.get("_last_llm_config") != current:
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=get_llm(provider, api_key),
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=True,
            )
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
        st.session_state.qa_chain = None
        st.session_state.indexed_files = []
        st.rerun()

# --------------------------------------------------------------------------
# Main chat area
# --------------------------------------------------------------------------
st.markdown('<p class="app-title">Chat</p>', unsafe_allow_html=True)

if not api_key:
    st.info("Enter your Groq API key in the sidebar to get started.")
elif not st.session_state.indexed_files:
    st.info("Upload at least one document (PDF, TXT, or CSV) in the sidebar to start chatting.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Send a message...")

if prompt:
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar first.")
    elif st.session_state.qa_chain is None:
        st.warning("Please upload a document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"query": prompt})
                answer = result["result"]
                sources = result.get("source_documents", [])

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
