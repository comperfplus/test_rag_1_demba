"""
Assistant vocal wolof + RAG sur documents PDF.
Déploiement Streamlit Cloud : définir OPENAI_API_KEY dans Settings > Secrets.
Déposer les PDF dans folder_rag/.
"""
import os
import logging
import tempfile
from pathlib import Path
from typing import List, Tuple
import itertools as it
import operator as op

import streamlit as st
import numpy as np
import tiktoken
import torch
import shutil
import subprocess
from openai import OpenAI

# Make FFmpeg discoverable by Transformers even when only imageio-ffmpeg is available.
try:
    import imageio_ffmpeg
    _ffmpeg_binary = Path(imageio_ffmpeg.get_ffmpeg_exe())
    _ffmpeg_bin_dir = Path(tempfile.gettempdir()) / "ffmpeg-bin"
    _ffmpeg_bin_dir.mkdir(parents=True, exist_ok=True)
    _ffmpeg_alias = _ffmpeg_bin_dir / "ffmpeg"
    if not _ffmpeg_alias.exists():
        try:
            _ffmpeg_alias.symlink_to(_ffmpeg_binary)
        except Exception:
            shutil.copy2(_ffmpeg_binary, _ffmpeg_alias)
            _ffmpeg_alias.chmod(0o755)
    os.environ["PATH"] = str(_ffmpeg_bin_dir) + os.pathsep + os.environ.get("PATH", "")
    logger = logging.getLogger("vocal_rag")
    logger.info("FFmpeg disponible : %s", shutil.which("ffmpeg"))
except Exception:
    logging.getLogger("vocal_rag").exception("Impossible de configurer FFmpeg depuis imageio-ffmpeg")

from PyPDF2 import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "folder_rag"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vocal_rag")

st.set_page_config(page_title="Assistant vocal wolof — RAG RGPH", page_icon="🗣️", layout="wide")
st.title("🗣️ Assistant vocal wolof — RAG sur les documents PDF")
st.write(
    "Enregistrez une question en wolof ou importez un fichier audio. "
    "L’application transcrit la question, la traduit en français, recherche dans les PDF, "
    "puis traduit la réponse en wolof."
)

# Streamlit Cloud: configure OPENAI_API_KEY under App settings > Secrets.
def get_api_key():
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass
    return key or os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = get_api_key()
if not OPENAI_API_KEY:
    st.error("La clé OPENAI_API_KEY est absente. Dans Streamlit Cloud, ouvre Settings > Secrets et ajoute : OPENAI_API_KEY = \"ta_cle_api\"")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 512
TOP_K = 5
TRANSLATION_CHECKPOINT = "galsenai/wolofToFrenchTranslator_nllb"
ASR_CHECKPOINT = "abdouaziz/wolof_wlarge"

@st.cache_resource(show_spinner="Chargement du modèle de traduction wolof/français…")
def load_translation_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_CHECKPOINT).to(device)
    model.eval()
    return tokenizer, model, device

@st.cache_resource(show_spinner="Chargement du modèle de reconnaissance vocale…")
def load_asr():
    return pipeline(
        "automatic-speech-recognition",
        model=ASR_CHECKPOINT,
        device=0 if torch.cuda.is_available() else -1,
    )

def translator(text: str, lang: str, tokenizer, model, device) -> str:
    if not text or not text.strip():
        return ""
    if lang.lower() == "wo":
        prefix = "translate Wolof to French: "
    elif lang.lower() == "fr":
        prefix = "translate French to Wolof: "
    else:
        raise ValueError("Le paramètre lang doit être 'wo' ou 'fr'.")
    inputs = tokenizer(prefix + text, return_tensors="pt").to(device)
    with torch.inference_mode():
        translated_tokens = model.generate(**inputs, max_length=230)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def load_pdf_pages(file_path: Path) -> List[str]:
    reader = PdfReader(str(file_path))
    return [(page.extract_text() or "") for page in reader.pages]

def chunk_pages(pages: List[str], chunk_size: int = CHUNK_SIZE) -> List[str]:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = list(it.chain.from_iterable(
        encoding.encode(page) for page in pages if page and page.strip()
    ))
    if not tokens:
        return []
    n_chunks = max(1, int(np.ceil(len(tokens) / chunk_size)))
    return [
        encoding.decode(list(part))
        for part in np.array_split(np.asarray(tokens), n_chunks)
        if len(part) > 0
    ]

def embed(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vectors = []
    for start in range(0, len(texts), 64):
        response = client.embeddings.create(
            input=texts[start:start + 64],
            model=EMBEDDING_MODEL,
        )
        vectors.extend(item.embedding for item in response.data)
    return np.asarray(vectors, dtype=np.float32)

def retrieve(query_embedding: np.ndarray, chunks: List[str], chunk_embeddings: np.ndarray, top_k: int = TOP_K) -> List[str]:
    if not chunks or chunk_embeddings.size == 0:
        return []
    denominator = np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    denominator = np.where(denominator == 0, 1e-12, denominator)
    scores = np.dot(chunk_embeddings, query_embedding) / denominator
    ranked = sorted(zip(chunks, scores), key=op.itemgetter(1), reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]

def ask(query: str, context_chunks: List[str]) -> str:
    context = "\n".join(f"- {chunk}" for chunk in context_chunks)
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content":
             "Tu es un assistant qui répond aux questions en se basant sur le contexte fourni, extrait des documents. "
             "Réponds clairement, naturellement et directement. Si le contexte ne contient pas la réponse, "
             "indique que l'information ne figure pas dans les documents et ne l'invente pas."},
            {"role": "user", "content": f"Contexte :\n{context}\n\nQuestion : {query}"},
        ],
        temperature=0,
    )
    return (completion.choices[0].message.content or "").strip()

@st.cache_resource(show_spinner="Lecture des PDF et indexation du corpus RAG…")
def build_corpus(pdf_paths: Tuple[str, ...], api_key: str):
    # Use an API client bound to the configured key in this cached operation.
    local_client = OpenAI(api_key=api_key)
    all_chunks = []
    for file_name in pdf_paths:
        pdf_file = Path(file_name)
        try:
            pages = load_pdf_pages(pdf_file)
            all_chunks.extend(chunk_pages(pages))
        except Exception:
            logger.exception("Impossible de traiter %s", pdf_file.name)
    if not all_chunks:
        raise ValueError("Aucun texte exploitable n’a été extrait des PDF. Vérifiez que les fichiers ne sont pas uniquement scannés.")
    vectors = []
    for start in range(0, len(all_chunks), 64):
        response = local_client.embeddings.create(input=all_chunks[start:start + 64], model=EMBEDDING_MODEL)
        vectors.extend(item.embedding for item in response.data)
    return all_chunks, np.asarray(vectors, dtype=np.float32)

pdf_files = tuple(str(p) for p in sorted(PDF_DIR.rglob("*.pdf"))) if PDF_DIR.exists() else tuple()
if not pdf_files:
    st.error(f"Aucun PDF trouvé dans `{PDF_DIR}`. Ajoutez les documents dans le dossier `folder_rag/` du dépôt GitHub.")
    st.stop()

try:
    with st.spinner("Préparation des modèles et des documents…"):
        tokenizer, translation_model, device = load_translation_model()
        asr = load_asr()
        corpus_chunks, corpus_embeddings = build_corpus(pdf_files, OPENAI_API_KEY)
except Exception as exc:
    logger.exception("Erreur pendant l'initialisation")
    st.error("L’application n’a pas pu charger les modèles ou les documents.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info("Consultez les logs Streamlit Cloud pour le détail. Vérifiez aussi la mémoire disponible et la clé OpenAI.")
    st.stop()

audio_file = st.audio_input("Enregistrez votre question en wolof")
uploaded_file = st.file_uploader("Ou importez un fichier audio", type=["wav", "mp3", "m4a", "ogg", "flac"])

audio_source = audio_file if audio_file is not None else uploaded_file
if st.button("Traiter la question", type="primary", disabled=(audio_source is None)):
    try:
        audio_bytes = audio_source.getvalue()
        suffix = Path(getattr(audio_source, "name", "question.wav")).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name
        try:
            with st.spinner("Transcription de l’audio…"):
                result = asr(audio_path)
            transcription = (result.get("text") or "").strip()
            if not transcription:
                st.warning("Je n’ai pas réussi à transcrire l’audio. Essayez un enregistrement plus clair.")
                st.stop()

            with st.spinner("Traduction et recherche dans les documents…"):
                traduction = translator(transcription, "wo", tokenizer, translation_model, device)
                query_embedding = embed([traduction])[0]
                top_chunks = retrieve(query_embedding, corpus_chunks, corpus_embeddings)
                if not top_chunks:
                    llm_response = "Je n’ai trouvé aucun passage exploitable dans les documents."
                else:
                    llm_response = ask(traduction, top_chunks)
                llm_reponse_wo = translator(llm_response, "fr", tokenizer, translation_model, device)

            st.subheader("Résultats")
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("Transcription wolof", transcription, height=130)
            with col2:
                st.text_area("Traduction française de la question", traduction, height=130)
            col3, col4 = st.columns(2)
            with col3:
                st.text_area("Réponse du chatbot en français", llm_response, height=200)
            with col4:
                st.text_area("Réponse du chatbot en wolof", llm_reponse_wo, height=200)
        finally:
            try:
                os.unlink(audio_path)
            except Exception:
                pass
    except Exception as exc:
        logger.exception("Erreur pendant le traitement audio")
        st.error(f"Une erreur est survenue pendant le traitement : {type(exc).__name__}: {exc}")
