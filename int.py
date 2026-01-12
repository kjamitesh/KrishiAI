# Author: Amitesh Jha | iSoft | 2025-10-07 (Refactored)
# Streamlit + LangChain RAG app — CPU-safe embeddings + Anthropic proxies-proof init.

# --- Stdlib ---
import os, glob, time, base64, hashlib, shutil, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# --- Third-party ---
import streamlit as st
import pandas as pd
from PIL import Image

# LangChain / loaders
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_community.document_loaders import (
    PyPDFLoader, BSHTMLLoader, Docx2txtLoader, CSVLoader, UnstructuredPowerPointLoader
)

# Ollama (chat first, fallback to LLM)
try:
    from langchain_community.chat_models import ChatOllama
except Exception:
    from langchain_community.llms import Ollama as ChatOllama

# Anthropic SDK (new & old)
try:
    from anthropic import Anthropic as _AnthropicClientNew
except Exception:
    _AnthropicClientNew = None
try:
    from anthropic import Client as _AnthropicClientOld
except Exception:
    _AnthropicClientOld = None


# --- Torch / device hygiene to avoid meta-tensor issues ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")            # force no CUDA
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

def _page_icon():
    p = Path("assets/llm.png")
    if p.exists():
        try:
            return Image.open(p)   # preferred: PIL image
        except Exception:
            return str(p)          # fallback: path string
    return "💬"                     # final fallback

st.set_page_config(
    page_title="AI Agent",
    page_icon=_page_icon(),
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Force-hide the sidebar & the collapse control ---
HIDE_SIDEBAR_CSS = """
<style>
section[data-testid="stSidebar"] { display: none !important; }
div[data-testid="stSidebar"] { display: none !important; }
button[kind="header"], [data-testid="collapsedControl"] { display: none !important; }
main.block-container { padding-left: 1rem; padding-right: 1rem; }
</style>
"""
st.markdown(HIDE_SIDEBAR_CSS, unsafe_allow_html=True)

# --------------------- Constants & Settings ---------------------
DEFAULT_OLLAMA = "llama3.2"
DEFAULT_CLAUDE = "claude-sonnet-4-5"  # stable public name
_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMB_MODEL_KW = {"device": "cpu", "trust_remote_code": False}
_ENCODE_KW = {"normalize_embeddings": True}

# --- Supported Extensions for Text/Documents ---
TEXT_EXTS = {".txt", ".md", ".rtf", ".html", ".htm", ".json", ".xml"}
DOC_EXTS  = {".pdf", ".docx", ".csv", ".tsv", ".pptx", ".pptm", ".doc", ".odt"}
SPREADSHEET_EXTS = {".xlsx", ".xlsm", ".xltx"}
SUPPORTED_TEXT_DOCS = TEXT_EXTS | DOC_EXTS | SPREADSHEET_EXTS

# --- Extensions for Media (placeholder only) ---
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a"}
VIDEO_EXTS = {".mp4", ".mov", ".avi"}
SUPPORTED_EXTS = SUPPORTED_TEXT_DOCS | IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS

GREETING_RE = re.compile(
    r"""^\s*(hi|hello|hey|hiya|yo|hola|namaste|namaskar|g'day|good\s+(morning|afternoon|evening))[\s!,.?]*$""",
    re.IGNORECASE,
)

# VectorStore type
VectorStoreType = FAISS

# --------------------- Minimal direct Claude model ---------------------
class ClaudeDirect(BaseChatModel):
    model: str = DEFAULT_CLAUDE
    temperature: float = 0.2
    max_tokens: int = 800
    _client: object = None

    def __init__(self, client, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_client", client)

    @property
    def _llm_type(self) -> str:
        return "claude_direct"

    def _convert_msgs(self, messages: list[BaseMessage]):
        out = []
        for m in messages:
            role = "user" if getattr(m, "type", "") == "human" else ("assistant" if getattr(m, "type", "") == "ai" else "user")
            if isinstance(m.content, str):
                text = m.content
            else:
                parts = m.content or []
                text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in parts)
            out.append({"role": role, "content": [{"type": "text", "text": text}]})
        return out

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        amsgs = self._convert_msgs(messages)
        resp = self._client.messages.create(
            model=self.model, messages=amsgs, temperature=self.temperature, max_tokens=self.max_tokens
        )
        text = ""
        content = getattr(resp, "content", []) or []
        for blk in content:
            if getattr(blk, "type", None) == "text":
                text += getattr(blk, "text", "") or ""
            elif isinstance(blk, dict) and blk.get("type") == "text":
                text += blk.get("text", "") or ""
        ai = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=ai)])

def build_citation_block(source_docs: List[Document], kb_root: str | None = None) -> str:
    # Citations intentionally suppressed per requirement.
    return ""

# --------------------- UI / THEME ---------------------
def _resolve_logo_path() -> Optional[Path]:
    env_logo = os.getenv("ISOFT_LOGO_PATH")
    candidates = [Path.cwd() / "assets" / "isoft_logo.png",
                  Path(env_logo).expanduser().resolve() if env_logo else None]
    for p in candidates:
        if p and p.exists():
            return p
    return None

def _first_existing(paths: list[Optional[Path]]) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None

def _resolve_avatar_paths() -> Tuple[Optional[Path], Optional[Path]]:
    user_env = os.getenv("USER_AVATAR_PATH")
    asst_env = os.getenv("ASSISTANT_AVATAR_PATH")
    user = _first_existing([
        Path(user_env).expanduser().resolve() if user_env else None,
        Path.cwd() / "assets" / "avatar.png",
    ])
    asst = _first_existing([
        Path(asst_env).expanduser().resolve() if asst_env else None,
        Path.cwd() / "assets" / "llm.png",
    ])
    return user, asst

def _img_to_data_uri(path: Optional[Path]) -> Optional[str]:
    if not path or not path.exists():
        return None
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    ext = (path.suffix.lower().lstrip(".") or "png")
    mime = "image/png" if ext in ("png", "apng") else ("image/jpeg" if ext in ("jpg", "jpeg") else "image/svg+xml")
    return f"data:{mime};base64,{b64}"

USER_AVATAR_PATH, ASSIST_AVATAR_PATH = _resolve_avatar_paths()
USER_AVATAR_URI = _img_to_data_uri(USER_AVATAR_PATH)
ASSIST_AVATAR_URI = _img_to_data_uri(ASSIST_AVATAR_PATH)
user_bg  = f"background-image:url('{USER_AVATAR_URI}');" if USER_AVATAR_URI else ""
asst_bg  = f"background-image:url('{ASSIST_AVATAR_URI}');" if ASSIST_AVATAR_URI else ""

css = f"""
<style>
:root{{ --bg:#f7f8fb; --sidebar-bg:#f5f7fb; --panel:#fff; --text:#0b1220;
       --muted:#5d6b82; --accent:#2563eb; --border:#e7eaf2;
       --bubble-user:#eef4ff; --bubble-assist:#f6f7fb; }}
html, body, [data-testid="stAppViewContainer"]{{ background:var(--bg); color:var(--text); }}
section[data-testid="stSidebar"]{{ background:var(--sidebar-bg); border-right:1px solid var(--border); }}
main .block-container{{ padding-top:.6rem; }}
.chat-card{{ background:var(--panel); border:1px solid var(--border); border-radius:14px; box-shadow:0 6px 16px rgba(16,24,40,.05); overflow:hidden; }}
.chat-scroll{{ max-height: 75vh; overflow:auto; padding:.65rem .9rem; }}
.msg{{ display:flex; align-items:flex-start; gap:.65rem; margin:.45rem 0; }}
.avatar{{ width:32px; height:32px; border-radius:50%; border:1px solid var(--border); background-size:cover; background-position:center; background-repeat:no-repeat; flex:0 0 32px; }}
.avatar.user {{ {user_bg} }}
.avatar.assistant {{ {asst_bg} }}
.bubble{{ border:1px solid var(--border); background:var(--bubble-assist); padding:.8rem .95rem; border-radius:12px; max-width:860px; white-space:pre-wrap; line-height:1.45; }}
.msg.user .bubble{{ background:var(--bubble-user); }}
.composer{{ padding:.6rem .75rem; border-top:1px solid var(--border); background:#fff; position:sticky; bottom:0; z-index:2; }}
.status-inline{{ width:100%; border:1px solid var(--border); background:#fafcff; border-radius:10px; padding:.5rem .7rem; font-size:.9rem; color:#111827; margin:.5rem 0 .8rem; }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# --------------------- Helpers ---------------------
def get_kb_dir() -> str:
    kb = os.path.abspath(os.path.join(".", "KB"))
    os.makedirs(kb, exist_ok=True)
    return kb

def human_time(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms/1000:.2f} s"

def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def iter_files(folder: str) -> List[str]:
    paths: List[str] = []
    for ext in SUPPORTED_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    return sorted(list(set(paths)))

def compute_kb_signature(folder: str) -> Tuple[str, int]:
    files = iter_files(folder)
    lines = []
    base = os.path.abspath(folder)
    for p in files:
        try:
            stt = os.stat(p)
            rel = os.path.relpath(os.path.abspath(p), base)
            lines.append(f"{rel}|{stt.st_size}|{int(stt.st_mtime)}")
        except Exception:
            continue
    lines.sort()
    raw = "\n".join(lines) + str(SUPPORTED_TEXT_DOCS)
    return stable_hash(raw if raw else f"EMPTY-{time.time()}"), len(files)

# --------------------- Loading ---------------------
def _fallback_read(path: str) -> str:
    try:
        if path.lower().endswith(tuple(SPREADSHEET_EXTS)):
            df = pd.read_excel(path).astype(str).iloc[:1000, :50]
            header = " | ".join(df.columns.tolist())
            body = "\n".join(" | ".join(row) for row in df.values.tolist())
            return f"Spreadsheet content from {Path(path).name}:\nColumns: {header}\nData:\n{body}"
        if path.lower().endswith((".csv", ".tsv")):
            sep = "\t" if path.lower().endswith(".tsv") else ","
            df = pd.read_csv(path, sep=sep).astype(str).iloc[:1000, :50]
            header = " | ".join(df.columns.tolist())
            body = "\n".join(" | ".join(row) for row in df.values.tolist())
            return f"CSV/TSV content from {Path(path).name}:\nColumns: {header}\nData:\n{body}"
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Error reading file {Path(path).name}: {e}")
        return ""

def load_one(path: str) -> List[Document]:
    p = path.lower()
    if p.endswith(tuple(IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS)):
        doc_type = "Image" if p.endswith(tuple(IMAGE_EXTS)) else ("Audio" if p.endswith(tuple(AUDIO_EXTS)) else "Video")
        placeholder_content = (
            f"This document is a {doc_type} file. "
            f"Text content unavailable (requires OCR/transcription). "
            f"Metadata: {Path(path).name}."
        )
        return [Document(page_content=placeholder_content, metadata={"source": path, "type": doc_type, "status": "placeholder"})]

    try:
        if p.endswith(".pdf"):
            return PyPDFLoader(path).load()
        if p.endswith((".html", ".htm")):
            return BSHTMLLoader(path).load()
        if p.endswith(".docx"):
            return Docx2txtLoader(path).load()
        if p.endswith((".pptx", ".pptm")):
            return UnstructuredPowerPointLoader(path).load()
        if p.endswith(".csv"):
            return CSVLoader(path).load()
        if p.endswith(".tsv"):
            return CSVLoader(path, csv_args={"delimiter": "\t"}).load()
        if p.endswith(tuple(TEXT_EXTS | SPREADSHEET_EXTS | {".doc", ".odt"})):
            txt = _fallback_read(path)
            return [Document(page_content=txt, metadata={"source": path})] if txt.strip() else []
        txt = _fallback_read(path)
        return [Document(page_content=txt, metadata={"source": path})] if txt.strip() else []
    except Exception as e:
        st.warning(f"Failed to load/process {Path(path).name} (Type: {p.split('.')[-1]}). Error: {e}")
        return []

def load_documents(folder: str) -> List[Document]:
    docs: List[Document] = []
    files_to_load = [p for p in iter_files(folder) if Path(p).suffix.lower() in SUPPORTED_EXTS]
    for path in files_to_load:
        docs.extend(load_one(path))
    return docs

# --------------------- Full-document helpers ---------------------
def _concat_docs(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "")
        page = meta.get("page")
        hdr = (f"\n\n--- [chunk {i} | page {page}] {Path(src).name} ---\n" if page is not None
               else f"\n\n--- [chunk {i}] {Path(src).name} ---\n")
        parts.append(hdr + (d.page_content or ""))
    return "".join(parts).strip()

def read_whole_file_from_disk(path: str) -> str:
    docs = load_one(path)
    return _concat_docs(docs)

def read_whole_doc_by_name(name_or_stem: str, base_folder: str) -> Tuple[str, List[str]]:
    name_or_stem = name_or_stem.lower().strip()
    candidates = [p for p in iter_files(base_folder) if name_or_stem in os.path.basename(p).lower()]
    texts = []
    for p in candidates:
        try:
            texts.append(read_whole_file_from_disk(p))
        except Exception as e:
            texts.append(f"[Error reading {os.path.basename(p)}: {e}]")
    return ("\n\n".join(t for t in texts if t.strip()) or ""), candidates

# --------------------- Indexing (FAISS) ---------------------
@dataclass
class ChunkingConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 200

def _make_embeddings():
    key = f"_emb_model_cache::{_EMB_MODEL}"
    if key in st.session_state:
        return st.session_state[key]
    embeddings = HuggingFaceEmbeddings(
        model_name=_EMB_MODEL,
        model_kwargs=_EMB_MODEL_KW,
        encode_kwargs=_ENCODE_KW,
    )
    st.session_state[key] = embeddings
    return embeddings

def _faiss_dir(persist_dir: str, collection_name: str) -> Path:
    return Path(persist_dir).expanduser().resolve() / collection_name

def index_folder_langchain(folder: str, persist_dir: str, collection_name: str,
                           emb_model: str, chunk_cfg: ChunkingConfig) -> Tuple[int, int]:
    raw_docs = load_documents(folder)

    # If nothing to index, clear any previous FAISS index cleanly
    faiss_dir = _faiss_dir(persist_dir, collection_name)
    if not raw_docs:
        if faiss_dir.exists():
            shutil.rmtree(faiss_dir, ignore_errors=True)
        return (0, 0)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg.chunk_size, chunk_overlap=chunk_cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    splat = splitter.split_documents(raw_docs)
    embeddings = _make_embeddings()
    faiss_db = FAISS.from_documents(documents=splat, embedding=embeddings)

    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss_db.save_local(str(faiss_dir))
    return (len(raw_docs), len(splat))

def get_vectorstore(persist_dir: str, collection_name: str, emb_model: str) -> Optional[FAISS]:
    key = f"_vs::{persist_dir}::{collection_name}::{emb_model}"
    if key in st.session_state:
        return st.session_state[key]

    faiss_path = _faiss_dir(persist_dir, collection_name)
    if not faiss_path.exists():
        return None

    try:
        vs = FAISS.load_local(
            folder_path=str(faiss_path),
            embeddings=_make_embeddings(),
            allow_dangerous_deserialization=True,
        )
        st.session_state[key] = vs
        return vs
    except Exception as e:
        st.error(f"Failed to load FAISS index from disk. Error: {e}")
        return None

# --------------------- Anthropic init helpers ---------------------
def _strip_proxy_env() -> None:
    for v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        os.environ.pop(v, None)

def _get_secret_api_key() -> Optional[str]:
    try:
        s = st.secrets
    except Exception:
        s = None

    if s:
        for k in ("ANTHROPIC_API_KEY","anthropic_api_key","CLAUDE_API_KEY","claude_api_key"):
            v = s.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for parent in ("anthropic","claude","secrets"):
            if parent in s and isinstance(s[parent], dict):
                ns = s[parent]
                for k in ("api_key","ANTHROPIC_API_KEY","key","token"):
                    v = ns.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
    for k in ("ANTHROPIC_API_KEY","anthropic_api_key","CLAUDE_API_KEY","claude_api_key"):
        v = os.getenv(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _anthropic_client_from_secrets():
    _strip_proxy_env()
    api_key = _get_secret_api_key()
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY (set in Streamlit secrets or env).")
    os.environ["ANTHROPIC_API_KEY"] = api_key
    if _AnthropicClientNew is not None:
        return _AnthropicClientNew(api_key=api_key)
    if _AnthropicClientOld is not None:
        return _AnthropicClientOld(api_key=api_key)
    raise RuntimeError("Anthropic SDK not installed correctly.")

# --------------------- Chain builders ---------------------
def make_llm(backend: str, model_name: str, temperature: float):
    if backend.startswith("Claude"):
        client = _anthropic_client_from_secrets()
        return ClaudeDirect(client=client, model=model_name or DEFAULT_CLAUDE,
                            temperature=temperature, max_tokens=800)
    return ChatOllama(model=model_name or DEFAULT_OLLAMA, temperature=temperature)

def make_chain(vs: VectorStoreType, llm: BaseChatModel, k: int):
    retriever = vs.as_retriever(search_kwargs={"k": k})
    memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory, return_source_documents=True, verbose=False
    )

# --------------------- Defaults + auto-index ---------------------
def settings_defaults() -> Dict[str, Any]:
    kb_dir = get_kb_dir()
    return {
        "persist_dir": ".faiss_index",
        "collection_name": f"kb-{stable_hash(kb_dir)}",
        "base_folder": kb_dir,
        "emb_model": _EMB_MODEL,
        "chunk_cfg": ChunkingConfig(),
        "backend": "Claude (Anthropic)",
        "ollama_model": DEFAULT_OLLAMA,
        "claude_model": DEFAULT_CLAUDE,
        "temperature": 0.2,
        "top_k": 5,
        "auto_index_min_interval_sec": 8,
    }

def auto_index_if_needed(status_placeholder: Optional[object] = None) -> Optional[VectorStoreType]:
    folder = st.session_state.get("base_folder")
    persist = st.session_state.get("persist_dir")
    colname = st.session_state.get("collection_name")
    emb_model = st.session_state.get("emb_model")
    min_gap = int(st.session_state.get("auto_index_min_interval_sec", 8))

    sig_now, file_count = compute_kb_signature(folder)
    last_sig = st.session_state.get("_kb_last_sig")
    last_time = float(st.session_state.get("_kb_last_index_ts", 0.0))
    now = time.time()

    need_index = (last_sig != sig_now) or (last_sig is None)
    throttled = (now - last_time) < min_gap
    target = status_placeholder if status_placeholder is not None else st

    faiss_path = _faiss_dir(persist, colname)
    index_exists = faiss_path.is_dir() and any(faiss_path.iterdir())

    if need_index and not throttled:
        try:
            target.markdown('<div class="status-inline">Indexing…</div>', unsafe_allow_html=True)
            n_docs, n_chunks = index_folder_langchain(folder, persist, colname, emb_model,
                                                      st.session_state.get("chunk_cfg", ChunkingConfig()))
            st.session_state["_kb_last_sig"] = sig_now
            st.session_state["_kb_last_index_ts"] = now
            st.session_state["_kb_last_counts"] = {"files": file_count, "docs": n_docs, "chunks": n_chunks}
            label = f"Indexed: <b>{n_docs}</b> files → <b>{n_chunks}</b> chunks"
        except Exception as e:
            label = f"Auto-index failed: <b>{e}</b>"
        target.markdown(f'<div class="status-inline">{label}</div>', unsafe_allow_html=True)
    elif not index_exists:
        try:
            target.markdown('<div class="status-inline">Index missing — building…</div>', unsafe_allow_html=True)
            n_docs, n_chunks = index_folder_langchain(folder, persist, colname, emb_model,
                                                      st.session_state.get("chunk_cfg", ChunkingConfig()))
            st.session_state["_kb_last_sig"] = sig_now
            st.session_state["_kb_last_index_ts"] = now
            st.session_state["_kb_last_counts"] = {"files": file_count, "docs": n_docs, "chunks": n_chunks}
            target.markdown(f'<div class="status-inline">Indexed: <b>{n_docs}</b> files → <b>{n_chunks}</b> chunks</div>',
                            unsafe_allow_html=True)
        except Exception as e:
            target.markdown(f'<div class="status-inline">Auto-index failed: <b>{e}</b></div>',
                            unsafe_allow_html=True)
    else:
        ts = st.session_state.get("_kb_last_index_ts")
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "—"
        target.markdown(
            f'<div class="status-inline">Auto-index is <b>ON</b> · Files: <b>{file_count}</b> · Last indexed: <b>{when}</b> · Index: <code>{colname}</code></div>',
            unsafe_allow_html=True
        )

    try:
        return get_vectorstore(persist, colname, emb_model)
    except Exception:
        return None

# --------------------- (Optional) Sidebar Settings (hidden by CSS) ---------------------
def render_sidebar():
    with st.sidebar:
        lp = _resolve_logo_path()
        if lp and Path(lp).exists():
            try:
                st.image(str(lp), caption="iSOFT ANZ Pvt Ltd", width=240)
            except Exception:
                pass
        else:
            st.info("Add assets/isoft_logo.png for branding.")

        st.subheader("⚙️ Settings")
        st.caption("Auto-index is enabled. Edit paths/models below if needed.")

        st.session_state["base_folder"] = st.text_input("Knowledge Base Folder", value=st.session_state["base_folder"])
        st.session_state["persist_dir"] = st.text_input("FAISS Persist Directory (Base)", value=st.session_state["persist_dir"])
        st.session_state["collection_name"] = st.text_input("FAISS Index Folder Name", value=st.session_state["collection_name"])

        st.divider()

        st.session_state["backend"] = st.radio("LLM Backend", ["Claude (Anthropic)", "Ollama (local)"], index=0, key="llm_backend_radio")
        if st.session_state["backend"].startswith("Claude"):
            st.session_state["claude_model"] = st.text_input("Claude Model Name", value=st.session_state["claude_model"])
        else:
            st.session_state["ollama_model"] = st.text_input("Ollama Model Name", value=st.session_state["ollama_model"])

        st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        st.session_state["top_k"] = st.slider("Top-K (Retrieval)", 1, 15, 5)
        st.session_state["auto_index_min_interval_sec"] = st.number_input("Auto-index min interval (sec)", 1, 300, 8, 1)

def _avatar_for_role(role: str) -> Optional[str]:
    if role == "user" and USER_AVATAR_PATH:
        return str(USER_AVATAR_PATH)
    if role == "assistant" and ASSIST_AVATAR_PATH:
        return str(ASSIST_AVATAR_PATH)
    return None

def render_chat_history():
    for message in st.session_state["messages"]:
        role = message["role"]
        with st.chat_message(role, avatar=_avatar_for_role(role)):
            st.markdown(message["content"])

# --------------------- Main Execution Logic ---------------------
def make_llm_and_chain(vs: VectorStoreType):
    backend = st.session_state["backend"]
    model_name = st.session_state["claude_model"] if backend.startswith("Claude") else st.session_state["ollama_model"]
    llm = make_llm(backend, model_name, float(st.session_state["temperature"]))
    chain = make_chain(vs, llm, int(st.session_state["top_k"]))
    return llm, chain, backend

def handle_user_input(query: str, vs: Optional[VectorStoreType]):
    st.session_state["messages"].append({"role": "user", "content": query})

    # Full-document commands
    m = re.match(r"^\s*(read|open|show)\s+(.+)$", query, flags=re.IGNORECASE)
    if m:
        target = m.group(2).strip().strip('"').strip("'")
        full_text, files = read_whole_doc_by_name(target, st.session_state["base_folder"])
        if not files:
            st.session_state["messages"].append({"role": "assistant", "content": f"Couldn't find a file containing “{target}” in the Knowledge Base folder."})
            st.rerun(); return

        if len(full_text) > 8000:
            try:
                llm, _, _ = make_llm_and_chain(vs or FAISS.from_texts([""], _make_embeddings()))
                summary = llm.invoke(f"Summarize the following document comprehensively:\n\n{full_text[:180000]}")
                # llm.invoke may return a str or a message-like object depending on model; normalize:
                summary_text = summary if isinstance(summary, str) else getattr(summary, "content", str(summary))
                reply = f"**Full-document summary for:** {', '.join(Path(p).name for p in files)}\n\n{summary_text}"
            except Exception as e:
                reply = f"Loaded the full document but failed to summarize: {e}\n\n--- RAW BEGIN ---\n{full_text[:20000]}\n--- RAW TRUNCATED ---"
        else:
            reply = f"**Full document content:**\n\n{full_text}"

        st.session_state["messages"].append({"role": "assistant", "content": reply})
        st.rerun(); return

    # Greetings → exactly "Hello"
    if GREETING_RE.match(query):
        st.session_state["messages"].append({"role": "assistant", "content": "Hello"})
        st.rerun(); return

    # Need vector store
    if vs is None:
        st.session_state["messages"].append({"role": "assistant", "content": "Vector store unavailable. Check your settings and ensure the FAISS index exists."})
        st.rerun(); return

    # RAG
    t0 = time.time()
    try:
        _, chain, backend = make_llm_and_chain(vs)
        with st.spinner(f"Querying {backend} with RAG..."):
            result = chain.invoke({"question": query})
            answer = result.get("answer", "").strip() or "I could not find an answer in the Knowledge Base."
            sources = result.get("source_documents", []) or []
        citation_block = build_citation_block(sources, kb_root=st.session_state.get("base_folder"))
        msg = f"{answer}{citation_block}\n\n_(Answered in {human_time((time.time()-t0)*1000)})_"
    except Exception as e:
        msg = f"RAG error: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": msg})
    st.rerun()
    
def header_with_icon(title: str = "AI Agent", icon_path: str = "assets/llm.png", height_px: int = 22):
    p = Path(icon_path)
    if p.exists():
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        icon_html = f"<img src='data:image/png;base64,{b64}' style='height:{height_px}px;vertical-align:-4px;margin-right:8px;'>"
    else:
        icon_html = "🤖 "  # fallback
    st.markdown(
        f"{icon_html}<span style='font-weight:600;font-size:1.15rem;'>{title}</span>",
        unsafe_allow_html=True,
    )
    
def main():
    for k, v in settings_defaults().items():
        st.session_state.setdefault(k, v)

    # Sidebar is rendered (for config) but hidden by CSS above
    render_sidebar()

    # --- Title with icon from assets/llm.png ---
    header_with_icon("AI Agent", icon_path="assets/llm.png", height_px=22)

    hero_status = st.container()
    vs = auto_index_if_needed(status_placeholder=hero_status)

    st.session_state.setdefault("messages", [
        {"role": "assistant", "content": "Hi! Ask anything about your Knowledge Base."}
    ])

    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
    render_chat_history()
    st.markdown("</div>", unsafe_allow_html=True)  # End chat-scroll

    st.markdown('<div class="composer">', unsafe_allow_html=True)
    user_text = st.chat_input("Type your question...", key="user_prompt_input")
    st.markdown("</div>", unsafe_allow_html=True)  # End composer
    st.markdown("</div>", unsafe_allow_html=True)  # End chat-card

    if user_text and user_text.strip():
        handle_user_input(user_text.strip(), vs)


    if user_text and user_text.strip():
        handle_user_input(user_text.strip(), vs)

if __name__ == "__main__":
    main()
