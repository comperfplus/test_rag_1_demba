# Assistant vocal wolof + RAG (Streamlit Cloud)

Cette version est adaptée à **Streamlit Cloud** (et non à Gradio/Hugging Face Spaces).

## Structure du dépôt
- `app.py` : application Streamlit.
- `requirements.txt` : dépendances Python.
- `folder_rag/` : placer ici les PDF à interroger.

## Déploiement sur Streamlit Cloud
1. Déposez ces fichiers dans votre dépôt GitHub.
2. Placez les documents PDF dans `folder_rag/` et poussez-les sur GitHub.
3. Dans Streamlit Cloud, choisissez le dépôt et le fichier principal `app.py`.
4. Dans **App settings > Secrets**, ajoutez :
   ```toml
   OPENAI_API_KEY = "votre_cle_api"
   ```
5. Redémarrez l’application.

## Fonctionnalités
- Enregistrement audio via le navigateur ou import d’un fichier audio.
- Transcription du wolof.
- Traduction de la question en français.
- Recherche RAG dans les PDF avec embeddings OpenAI.
- Génération d’une réponse en français puis traduction en wolof.

## Points d’attention
- Le premier démarrage télécharge les modèles Hugging Face et peut être long.
- Les modèles vocaux peuvent dépasser la mémoire disponible sur une instance Streamlit Cloud gratuite.
- Les PDF scannés sans couche texte nécessitent une étape OCR.
- Ne publiez jamais votre clé OpenAI dans GitHub.
