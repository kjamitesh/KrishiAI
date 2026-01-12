# Author: Amitesh Jha 
# Streamlit + LangChain RAG app — OpenAI-first (LLM + embeddings), CPU-safe indexing.
#
# Change in this version:
# - Sidebar removed completely (no sidebar code, no render call) without changing app behavior.

# --- Stdlib ---
import os, glob, time, base64, hashlib, shutil, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# --- Third-party ---
import streamlit as st
import pandas as pd
from PIL import Image

# LangChain / loaders / vectorstore
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# OpenAI via LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.document_loaders import (
    PyPDFLoader, BSHTMLLoader, Docx2txtLoader, CSVLoader, UnstructuredPowerPointLoader
)

# --------------------- Torch / device hygiene ---------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# --------------------- Page config ---------------------
def _page_icon():
    p = Path("assets/llm.png")
    if p.exists():
        try:
            return Image.open(p)
        except Exception:
            return str(p)
    return "💬"

st.set_page_config(
    page_title="AI Krishi Agent",
    page_icon=_page_icon(),
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Hide the sidebar & collapse control (kept; harmless even when sidebar unused) ---
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
DEFAULT_OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

TEXT_EXTS = {".txt", ".md", ".rtf", ".html", ".htm", ".json", ".xml"}
DOC_EXTS  = {".pdf", ".docx", ".csv", ".tsv", ".pptx", ".pptm", ".doc", ".odt"}
SPREADSHEET_EXTS = {".xlsx", ".xlsm", ".xltx"}
SUPPORTED_TEXT_DOCS = TEXT_EXTS | DOC_EXTS | SPREADSHEET_EXTS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a"}
VIDEO_EXTS = {".mp4", ".mov", ".avi"}
SUPPORTED_EXTS = SUPPORTED_TEXT_DOCS | IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS

GREETING_RE = re.compile(
    r"""^\s*(hi|hello|hey|hiya|yo|hola|namaste|namaskar|g'day|good\s+(morning|afternoon|evening))[\s!,.?]*$""",
    re.IGNORECASE,
)

VectorStoreType = FAISS

# --------------------- Citations ---------------------
def build_citation_block(source_docs: List[Document], kb_root: str | None = None) -> str:
    return ""

# --------------------- UI / THEME ---------------------
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
:root{{ --bg:#f7f8fb; --panel:#fff; --text:#0b1220;
       --muted:#5d6b82; --accent:#2563eb; --border:#e7eaf2;
       --bubble-user:#eef4ff; --bubble-assist:#f6f7fb; }}
html, body, [data-testid="stAppViewContainer"]{{ background:var(--bg); color:var(--text); }}
main .block-container{{ padding-top:.6rem; }}
.chat-card{{ background:var(--panel); border:1px solid var(--border); border-radius:14px; box-shadow:0 6px 16px rgba(16,24,40,.05); overflow:hidden; }}
.chat-scroll{{ max-height: 75vh; overflow:auto; padding:.65rem .9rem; }}
.status-inline{{ width:100%; border:1px solid var(--border); background:#fafcff; border-radius:10px; padding:.5rem .7rem; font-size:.9rem; color:#111827; margin:.5rem 0 .8rem; }}
.composer{{ padding:.6rem .75rem; border-top:1px solid var(--border); background:#fff; position:sticky; bottom:0; z-index:2; }}
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

def _ensure_openai_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    try:
        k = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_api_key")
    except Exception:
        k = None
    if k and isinstance(k, str) and k.strip():
        os.environ["OPENAI_API_KEY"] = k.strip()

def _make_embeddings() -> OpenAIEmbeddings:
    key = f"_emb_model_cache::openai::{DEFAULT_OPENAI_EMBED_MODEL}"
    if key in st.session_state:
        return st.session_state[key]
    _ensure_openai_key()
    embeddings = OpenAIEmbeddings(model=DEFAULT_OPENAI_EMBED_MODEL)
    st.session_state[key] = embeddings
    return embeddings

def _faiss_dir(persist_dir: str, collection_name: str) -> Path:
    return Path(persist_dir).expanduser().resolve() / collection_name

def index_folder_langchain(folder: str, persist_dir: str, collection_name: str,
                           chunk_cfg: ChunkingConfig) -> Tuple[int, int]:
    raw_docs = load_documents(folder)

    faiss_dir = _faiss_dir(persist_dir, collection_name)
    if not raw_docs:
        if faiss_dir.exists():
            shutil.rmtree(faiss_dir, ignore_errors=True)
        return (0, 0)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg.chunk_size,
        chunk_overlap=chunk_cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    splat = splitter.split_documents(raw_docs)

    embeddings = _make_embeddings()
    faiss_db = FAISS.from_documents(documents=splat, embedding=embeddings)

    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss_db.save_local(str(faiss_dir))
    return (len(raw_docs), len(splat))

def get_vectorstore(persist_dir: str, collection_name: str) -> Optional[FAISS]:
    key = f"_vs::{persist_dir}::{collection_name}::openai::{DEFAULT_OPENAI_EMBED_MODEL}"
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

# --------------------- Chain builders ---------------------
def make_llm(model_name: str, temperature: float) -> ChatOpenAI:
    _ensure_openai_key()
    return ChatOpenAI(model=model_name or DEFAULT_OPENAI_CHAT_MODEL, temperature=temperature)

def make_chain(vs: VectorStoreType, llm: ChatOpenAI, k: int):
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
        "chunk_cfg": ChunkingConfig(),
        "openai_chat_model": DEFAULT_OPENAI_CHAT_MODEL,
        "temperature": 0.2,
        "top_k": 5,
        "auto_index_min_interval_sec": 8,
    }

def auto_index_if_needed(status_placeholder: Optional[object] = None) -> Optional[VectorStoreType]:
    folder = st.session_state.get("base_folder")
    persist = st.session_state.get("persist_dir")
    colname = st.session_state.get("collection_name")
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
            n_docs, n_chunks = index_folder_langchain(
                folder, persist, colname, st.session_state.get("chunk_cfg", ChunkingConfig())
            )
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
            n_docs, n_chunks = index_folder_langchain(
                folder, persist, colname, st.session_state.get("chunk_cfg", ChunkingConfig())
            )
            st.session_state["_kb_last_sig"] = sig_now
            st.session_state["_kb_last_index_ts"] = now
            st.session_state["_kb_last_counts"] = {"files": file_count, "docs": n_docs, "chunks": n_chunks}
            target.markdown(
                f'<div class="status-inline">Indexed: <b>{n_docs}</b> files → <b>{n_chunks}</b> chunks</div>',
                unsafe_allow_html=True
            )
        except Exception as e:
            target.markdown(f'<div class="status-inline">Auto-index failed: <b>{e}</b></div>', unsafe_allow_html=True)

    else:
        ts = st.session_state.get("_kb_last_index_ts")
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "—"
        target.markdown(
            f'<div class="status-inline">Auto-index is <b>ON</b> · Files: <b>{file_count}</b> · Last indexed: <b>{when}</b> · Index: <code>{colname}</code></div>',
            unsafe_allow_html=True
        )

    try:
        return get_vectorstore(persist, colname)
    except Exception:
        return None

# --------------------- Chat UI helpers ---------------------
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

def make_llm_and_chain(vs: VectorStoreType):
    model_name = st.session_state["openai_chat_model"]
    llm = make_llm(model_name, float(st.session_state["temperature"]))
    chain = make_chain(vs, llm, int(st.session_state["top_k"]))
    return llm, chain, model_name

def handle_user_input(query: str, vs: Optional[VectorStoreType]):
    st.session_state["messages"].append({"role": "user", "content": query})

    # Full-document commands
    m = re.match(r"^\s*(read|open|show)\s+(.+)$", query, flags=re.IGNORECASE)
    if m:
        target = m.group(2).strip().strip('"').strip("'")
        full_text, files = read_whole_doc_by_name(target, st.session_state["base_folder"])
        if not files:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"Couldn't find a file containing “{target}” in the Knowledge Base folder."
            })
            st.rerun()
            return

        if len(full_text) > 8000:
            try:
                llm, _, model_name = make_llm_and_chain(vs or FAISS.from_texts([""], _make_embeddings()))
                summary = llm.invoke(f"Summarize the following document comprehensively:\n\n{full_text[:180000]}")
                summary_text = summary if isinstance(summary, str) else getattr(summary, "content", str(summary))
                reply = f"**Full-document summary for:** {', '.join(Path(p).name for p in files)}\n\n{summary_text}\n\n_(Model: {model_name})_"
            except Exception as e:
                reply = (
                    f"Loaded the full document but failed to summarize: {e}\n\n"
                    f"--- RAW BEGIN ---\n{full_text[:20000]}\n--- RAW TRUNCATED ---"
                )
        else:
            reply = f"**Full document content:**\n\n{full_text}"

        st.session_state["messages"].append({"role": "assistant", "content": reply})
        st.rerun()
        return

    # Greetings → exactly "Hello"
    if GREETING_RE.match(query):
        st.session_state["messages"].append({"role": "assistant", "content": "Hello"})
        st.rerun()
        return

    # Need vector store
    if vs is None:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "Vector store unavailable. Check your settings and ensure the FAISS index exists."
        })
        st.rerun()
        return

    # RAG
    t0 = time.time()
    try:
        _, chain, model_name = make_llm_and_chain(vs)
        with st.spinner(f"Querying OpenAI ({model_name}) with RAG..."):
            result = chain.invoke({"question": query})
            answer = result.get("answer", "").strip() or "I could not find an answer in the Knowledge Base."
            sources = result.get("source_documents", []) or []
        citation_block = build_citation_block(sources, kb_root=st.session_state.get("base_folder"))
        msg = f"{answer}{citation_block}\n\n_(Answered in {human_time((time.time()-t0)*1000)})_"
    except Exception as e:
        msg = f"RAG error: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": msg})
    st.rerun()

def header_with_icon(title: str = "AI Krishi Agent", icon_path: str = "assets/llm.png", height_px: int = 22):
    p = Path(icon_path)
    if p.exists():
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        icon_html = f"<img src='data:image/png;base64,{b64}' style='height:{height_px}px;vertical-align:-4px;margin-right:8px;'>"
    else:
        icon_html = "🤖 "
    st.markdown(
        f"{icon_html}<span style='font-weight:600;font-size:1.15rem;'>{title}</span>",
        unsafe_allow_html=True,
    )

def main():
    for k, v in settings_defaults().items():
        st.session_state.setdefault(k, v)

    header_with_icon("AI Krishi Agent", icon_path="assets/llm.png", height_px=22)

    hero_status = st.container()
    vs = auto_index_if_needed(status_placeholder=hero_status)

    st.session_state.setdefault("messages", [
        {"role": "assistant", "content": "Hi! Ask anything about Krishi"}
    ])

    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
    render_chat_history()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="composer">', unsafe_allow_html=True)
    user_text = st.chat_input("Type your question...", key="user_prompt_input")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if user_text and user_text.strip():
        handle_user_input(user_text.strip(), vs)

if __name__ == "__main__":
    main()
