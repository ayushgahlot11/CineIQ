# CineIQ: Hybrid Explainable Recommendation Engine

An end-to-end, decoupled machine learning system that provides personalized, explainable movie recommendations. CineIQ moves beyond traditional recommendation loops by combining matrix factorization, content similarity, and deep-learning sentiment analysis.

## The Architecture
CineIQ uses a decoupled architecture for scalable inference:
* **Frontend:** A dynamic Streamlit dashboard featuring interactive Plotly radar charts for user taste profiling.
* **Backend:** A FastAPI REST API serving multiple ML models from memory for low-latency inference.
* **Tracking:** MLflow integration for model training and experiment tracking.

## The Machine Learning Engine (Hybrid Approach)
1. **Collaborative Filtering:** Uses the `scikit-surprise` SVD algorithm to predict user ratings based on latent behavioral patterns in the MovieLens 25M dataset.
2. **Content-Based Filtering:** Uses Scikit-learn's `TfidfVectorizer` and Cosine Similarity on TMDB metadata (Genres, Keywords, Cast, Director) to find thematic matches.
3. **Sentiment Re-Ranking:** Leverages a custom-finetuned **HuggingFace DistilBERT** transformer model (trained via Google Colab T4 GPU on 50k IMDB reviews) to analyze metadata sentiment and mathematically adjust final recommendation scores based on audience reception.

## Explainable AI (XAI)
CineIQ features an explainability layer that generates a human-readable justification for every single recommendation, combining SVD matrix predictions with DistilBERT's textual sentiment evaluation.

## Tech Stack
* **Machine Learning:** Scikit-learn, Surprise (SVD), Pandas, NumPy
* **Deep Learning / NLP:** HuggingFace Transformers (DistilBERT), PyTorch
* **Serving & UI:** FastAPI, Uvicorn, Streamlit, Plotly
* **DevOps / Tracking:** MLflow, Git
