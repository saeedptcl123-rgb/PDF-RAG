import os
import re
from io import BytesIO

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Assistant")
st.caption(
    "Upload a PDF, build a local FAISS vector index, and ask questions using "
    "the open-weight GPT-OSS 120B model through Groq."
)


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        return None
    return Groq(api_key=api_key)


# -----------------------------
# PDF -> text
# -----------------------------
def extract_pdf_text(pdf_bytes: bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# -----------------------------
# Text -> chunks
# -----------------------------
def create_chunks(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []

    for page in pages:
        text = page["text"]

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page["page"],
                    }
                )

            if end >= len(text):
                break

            start = max(0, end - overlap)

    return chunks


# -----------------------------
# Chunks -> embeddings -> FAISS
# -----------------------------
def build_faiss_index(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner product + normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


# -----------------------------
# Retrieval
# -----------------------------
def retrieve(query, model, index, chunks, top_k=TOP_K):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = chunks[int(idx)].copy()
        item["score"] = float(score)
        results.append(item)

    return results


# -----------------------------
# Groq generation
# -----------------------------
def generate_answer(question, retrieved_chunks):
    client = get_groq_client()

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets or "
            "set it as an environment variable."
        )

    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i} | PDF page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a careful PDF question-answering assistant.

Answer the user's question using ONLY the supplied PDF context.
If the answer is not contained in the context, say:
"I could not find that information in the uploaded PDF."

Do not invent facts or citations.
When possible, mention the PDF page number(s) that support the answer.
Keep the answer clear and reasonably concise.
"""

    user_prompt = f"""PDF CONTEXT:

{context}

USER QUESTION:
{question}

Answer based only on the PDF context above.
"""

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1200,
    )

    return completion.choices[0].message.content


# -----------------------------
# Session state
# -----------------------------
if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "filename" not in st.session_state:
    st.session_state.filename = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=TOP_K,
    )

    st.markdown("---")
    st.write("**LLM:**", LLM_MODEL)
    st.write("**Embedding:**", EMBEDDING_MODEL)
    st.write("**Vector DB:** FAISS")
    st.write("**Chunk size:**", CHUNK_SIZE)
    st.write("**Chunk overlap:**", CHUNK_OVERLAP)

    if st.session_state.filename:
        st.success(f"Indexed: {st.session_state.filename}")

    if st.button("Clear current document"):
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.filename = None
        st.session_state.messages = []
        st.rerun()


# -----------------------------
# Upload + indexing
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="The PDF is extracted and indexed in memory for this Streamlit session.",
)

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.filename:
        pdf_bytes = uploaded_file.getvalue()

        with st.status("Building RAG index...", expanded=True) as status:
            st.write("1. Extracting PDF text...")
            pages = extract_pdf_text(pdf_bytes)

            if not pages:
                status.update(label="No extractable text found", state="error")
                st.error(
                    "No text could be extracted from this PDF. "
                    "If it is a scanned/image-only PDF, OCR is required."
                )
                st.stop()

            st.write(f"2. Extracted text from {len(pages)} page(s).")

            st.write("3. Creating chunks...")
            chunks = create_chunks(pages)

            if not chunks:
                status.update(label="No chunks created", state="error")
                st.error("No usable text chunks were created.")
                st.stop()

            st.write(f"4. Created {len(chunks)} chunks.")

            st.write("5. Creating embeddings and FAISS index...")
            embedding_model = load_embedding_model()
            index = build_faiss_index(chunks, embedding_model)

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.filename = uploaded_file.name
            st.session_state.messages = []

            status.update(
                label=f"RAG index ready — {len(chunks)} chunks",
                state="complete",
            )

    if st.session_state.index is not None:
        st.success(
            f"Ready to answer questions about **{st.session_state.filename}**."
        )

# -----------------------------
# Chat history
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Retrieved sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**Page {source['page']} — similarity {source['score']:.3f}**"
                    )
                    st.write(source["text"])


# -----------------------------
# Question answering
# -----------------------------
question = st.chat_input(
    "Ask a question about your uploaded PDF..."
)

if question:
    if st.session_state.index is None:
        st.warning("Please upload and index a PDF first.")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            embedding_model = load_embedding_model()

            with st.spinner("Searching the PDF..."):
                retrieved = retrieve(
                    question,
                    embedding_model,
                    st.session_state.index,
                    st.session_state.chunks,
                    top_k=top_k,
                )

            with st.spinner("Generating answer with Groq..."):
                answer = generate_answer(question, retrieved)

            st.markdown(answer)

            with st.expander("Retrieved sources"):
                for source in retrieved:
                    st.markdown(
                        f"**Page {source['page']} — similarity {source['score']:.3f}**"
                    )
                    st.write(source["text"])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved,
                }
            )

        except Exception as exc:
            st.error(f"Error: {exc}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "RAG pipeline: PDF → extraction → chunks → embeddings → FAISS retrieval → Groq GPT-OSS."
)
