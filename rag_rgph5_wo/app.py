"""
Assistant vocal wolof + RAG sur documents PDF.
Lancer avec : python app.py
Les documents PDF doivent être placés dans ./folder_rag/
La clé OpenAI doit être définie dans un fichier .env ou dans l'environnement.
"""

import spaces

import os
import logging
from pathlib import Path
from typing import List, Tuple
import itertools as it
import operator as op

import gradio as gr
import numpy as np
import tiktoken
import torch
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "folder_rag"
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vocal_rag")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY est manquante. Crée un fichier .env à côté de app.py "
        "et ajoute : OPENAI_API_KEY=ta_cle_api"
    )

client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 512
TOP_K = 5

TRANSLATION_CHECKPOINT = "galsenai/wolofToFrenchTranslator_nllb"
ASR_CHECKPOINT = "abdouaziz/wolof_wlarge"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Périphérique de calcul : %s", device)

# ---------------------------------------------------------------------
# Modèle de traduction Wolof <-> Français
# ---------------------------------------------------------------------
logger.info("Chargement du modèle de traduction : %s", TRANSLATION_CHECKPOINT)
translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_CHECKPOINT)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_CHECKPOINT
).to(device)
translation_model.eval()
 

def translator(text: str, lang: str = "wo") -> str:
    """Traduit du wolof vers le français (wo) ou du français vers le wolof (fr)."""
    if not text or not text.strip():
        return ""
    if lang.lower() == "wo":
        prefix = "translate Wolof to French: "
    elif lang.lower() == "fr":
        prefix = "translate French to Wolof: "
    else:
        raise ValueError("Le paramètre lang doit être 'wo' ou 'fr'.")

    inputs = translation_tokenizer(prefix + text, return_tensors="pt").to(device)
    with torch.inference_mode():
        translated_tokens = translation_model.generate(**inputs, max_length=230)
    return translation_tokenizer.batch_decode(
        translated_tokens, skip_special_tokens=True
    )[0]


# ---------------------------------------------------------------------
# RAG : lecture des PDF, découpage, embeddings et recherche
# ---------------------------------------------------------------------
def load_pdf_pages(file_path: Path) -> List[str]:
    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages


def chunk_pages(pages: List[str], chunk_size: int = CHUNK_SIZE) -> List[str]:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = list(it.chain.from_iterable(
        encoding.encode(page) for page in pages if page and page.strip()
    ))
    if not tokens:
        return []
    number_of_chunks = max(1, int(np.ceil(len(tokens) / chunk_size)))
    return [
        encoding.decode(list(part))
        for part in np.array_split(np.asarray(tokens), number_of_chunks)
        if len(part) > 0
    ]


def embed(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    # Appel par lots pour éviter d'envoyer trop de textes en une seule requête.
    vectors = []
    batch_size = 64
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(
            input=texts[start:start + batch_size],
            model=EMBEDDING_MODEL,
        )
        vectors.extend(item.embedding for item in response.data)
    return np.asarray(vectors, dtype=np.float32)


def retrieve(
    query_embedding: np.ndarray,
    chunks: List[str],
    chunk_embeddings: np.ndarray,
    top_k: int = TOP_K,
) -> List[str]:
    if not chunks or chunk_embeddings.size == 0:
        return []
    denominator = (
        np.linalg.norm(chunk_embeddings, axis=1)
        * np.linalg.norm(query_embedding)
    )
    denominator = np.where(denominator == 0, 1e-12, denominator)
    scores = np.dot(chunk_embeddings, query_embedding) / denominator
    ranked = sorted(zip(chunks, scores), key=op.itemgetter(1), reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]


def ask(query: str, context_chunks: List[str]) -> str:
    context = "\n".join(f"- {chunk}" for chunk in context_chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant qui répond aux questions en se basant sur le "
                "contexte fourni, extrait des documents. Réponds clairement, "
                "naturellement et directement. Si le contexte ne contient pas la "
                "réponse, indique que l'information ne figure pas dans les documents "
                "et ne l'invente pas."
            ),
        },
        {
            "role": "user",
            "content": f"Contexte :\n{context}\n\nQuestion : {query}",
        },
    ]
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,
    )
    return completion.choices[0].message.content.strip()


def build_corpus(pdf_dir: Path) -> Tuple[List[str], np.ndarray]:
    pdf_files = sorted(pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"Aucun PDF trouvé dans {pdf_dir}. Ajoute tes documents PDF dans ce dossier."
        )

    all_chunks = []
    for pdf_file in pdf_files:
        logger.info("Lecture du PDF : %s", pdf_file.name)
        try:
            pages = load_pdf_pages(pdf_file)
            file_chunks = chunk_pages(pages)
            all_chunks.extend(file_chunks)
            logger.info("  %d segments extraits", len(file_chunks))
        except Exception:
        # Un document problématique ne doit pas empêcher les autres d'être lus.
            logger.exception("Impossible de traiter %s", pdf_file)
    if not all_chunks:
        raise ValueError(
            "Aucun texte exploitable n'a été extrait des PDF. "
            "Vérifie qu'ils contiennent du texte et ne sont pas uniquement scannés."
        )
    logger.info("Calcul des embeddings pour %d segments...", len(all_chunks))
    return all_chunks, embed(all_chunks)


logger.info("Construction de la base documentaire depuis %s", PDF_DIR)
corpus_chunks, corpus_embeddings = build_corpus(PDF_DIR)
logger.info("Base RAG prête : %d segments", len(corpus_chunks))


def get_response_from_llm(question: str) -> str:
    query_embedding = embed([question])[0]
    top_chunks = retrieve(query_embedding, corpus_chunks, corpus_embeddings)
    if not top_chunks:
        return "Je n'ai trouvé aucun passage exploitable dans les documents."
    return ask(question, top_chunks)


# ---------------------------------------------------------------------
# Reconnaissance vocale (ASR)
# ---------------------------------------------------------------------
logger.info("Chargement du modèle de reconnaissance vocale : %s", ASR_CHECKPOINT)
asr = pipeline(
    "automatic-speech-recognition",
    model=ASR_CHECKPOINT,
    device=0 if torch.cuda.is_available() else -1,
)

@spaces.GPU(duration=120)
def audio_to_text(audio_path):
    if audio_path is None:
        return "", "", "", "Veuillez enregistrer une question avec le microphone."

    try:
        result = asr(audio_path)
        transcription = result.get("text", "").strip()
        if not transcription:
            return "", "", "", "Je n'ai pas réussi à transcrire l'audio."

        traduction = translator(text=transcription, lang="wo")
        llm_response = get_response_from_llm(traduction)
        llm_reponse_wo = translator(text=llm_response, lang="fr")

        return transcription, traduction, llm_response, llm_reponse_wo
    except Exception as exc:
        logger.exception("Erreur pendant le traitement audio")
        return "", "", "", f"Une erreur est survenue : {exc}"


# ---------------------------------------------------------------------
# Interface Gradio
# ---------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container textarea,
.gradio-container input {
    border: 2px solid #2563eb !important;
    border-radius: 10px !important;
}

.gradio-container textarea:focus,
.gradio-container input:focus {
    border-color: #111827 !important;
    box-shadow: 0 0 0 1px #111827 !important;
}

.output-grid {
    gap: 16px !important;
}

/* Deux derniers champs : contours verts */
.gradio-container .output-grid:nth-of-type(2) textarea {
    border: 2px solid #16a34a !important;
    border-radius: 10px !important;
}

/* Contour vert foncé au focus */
.gradio-container .output-grid:nth-of-type(2) textarea:focus {
    border-color: #166534 !important;
    box-shadow: 0 0 0 1px #166534 !important;
}
"""





with gr.Blocks(
    title="Assistant vocal Wolof - RAG RGPH",
    css=CUSTOM_CSS + "\nfooter {visibility: hidden;}",
) as demo:
    gr.Markdown("# Assistant vocal wolof — RAG sur les documents PDF")
    gr.Markdown(
        "Posez votre question en wolof avec le microphone. "
        "L'application transcrit la question, la traduit en français, "
        "recherche dans les PDF, puis traduit la réponse en wolof."
    )

    with gr.Tab("Assistant vocal"):
       # audio_input = gr.Audio(
       #     sources=["microphone"],
       #     type="filepath",
       #     label="Enregistrer votre question en wolof",
       # )


        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Enregistrez votre question ou importez un fichier audio",
            format="wav",
        )


        submit_btn = gr.Button("Soumettre", variant="primary")
        #transcription_output = gr.Textbox(label="Transcription wolof", lines=3)
        #translation_output = gr.Textbox(label="Traduction française de la question", lines=3)
        #answer_fr_output = gr.Textbox(label="Réponse du chatbot en français", lines=5)
        #answer_wo_output = gr.Textbox(label="Réponse du chatbot en wolof", lines=5)
        
        with gr.Row(elem_classes=["output-grid"]):
            transcription_output = gr.Textbox(
                label="Transcription wolof",
                lines=3,
                elem_classes=["output-field"],
            )
            translation_output = gr.Textbox(
                label="Traduction française de la question",
                lines=3,
                elem_classes=["output-field"],
            )

        with gr.Row(elem_classes=["output-grid", "response-green"]):
            answer_fr_output = gr.Textbox(
                label="Réponse du chatbot en français",
                lines=5,
                elem_classes=["output-field"],
            )
            answer_wo_output = gr.Textbox(
                label="Réponse du chatbot en wolof",
                lines=5,
                elem_classes=["output-field"],
            )




        submit_btn.click(
            fn=audio_to_text,
            inputs=[audio_input],
            outputs=[
                transcription_output,
                translation_output,
                answer_fr_output,
                answer_wo_output,
            ],
        )

if __name__ == "__main__":
    demo.launch(
        debug=False,
        #server_name="127.0.0.1",
         server_name="0.0.0.0",
        ssr_mode=False,
    )
