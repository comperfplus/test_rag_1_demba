# Assistant vocal wolof + RAG (version locale)

## Structure
- `app.py` : application Gradio.
- `requirements.txt` : dépendances Python.
- `.env.example` : modèle de configuration de la clé OpenAI.
- `folder_rag/` : déposer ici tous les PDF à indexer.

## Installation
1. Installer Python 3.10 ou 3.11.
2. Ouvrir un terminal dans ce dossier.
3. Créer un environnement virtuel :
   - Windows : `py -m venv .venv` puis `.venv\\Scripts\\activate`
   - Linux/macOS : `python3 -m venv .venv` puis `source .venv/bin/activate`
4. Installer les dépendances : `pip install -r requirements.txt`
5. Copier `.env.example` vers `.env`, puis renseigner `OPENAI_API_KEY`.
6. Copier tous les PDF dans `folder_rag/`.
7. Lancer : `python app.py`
8. Ouvrir l'adresse locale indiquée par Gradio, généralement `http://127.0.0.1:7860`.

## GPU
Le code utilise CUDA si une version de PyTorch compatible GPU est installée et si un GPU NVIDIA est disponible. Sinon, il utilise le CPU, ce qui peut être très lent pour les modèles vocaux. Pour une installation GPU, installer PyTorch selon les instructions officielles correspondant à votre version de CUDA : https://pytorch.org/get-started/locally/

## Sécurité
La clé OpenAI doit rester dans `.env` et ne doit jamais être publiée. Le dossier `.env` n'est pas inclus dans cette archive.
Les PDF scannés sans couche texte nécessitent une étape OCR, non incluse ici.
