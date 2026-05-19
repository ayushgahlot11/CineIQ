from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

app = FastAPI(title="CineIQ Hybrid Engine API", version="1.0")

print("Booting up FastAPI and loading models into memory...")

base_dir = "../artifacts/saved_models/"
with open(os.path.join(base_dir, 'svd_model.pkl'), 'rb') as f:
    svd_model = pickle.load(f)
with open(os.path.join(base_dir, 'tfidf_matrix.pkl'), 'rb') as f:
    tfidf_matrix = pickle.load(f)
with open(os.path.join(base_dir, 'movie_indices.pkl'), 'rb') as f:
    indices = pickle.load(f)

metadata = pd.read_csv("../data/processed/master_metadata.csv")

# --- CUSTOM DISTILBERT INITIALIZATION ---
# This will crash IF the folder isn't there yet, which is expected!
# It is waiting for your Colab download.
print("Loading Custom DistilBERT Sentiment Model from local artifacts...")
model_path = "../artifacts/distilbert_imdb"

try:
    sentiment_analyzer = pipeline("text-classification", model=model_path, tokenizer=model_path)
except Exception as e:
    print(f"⚠️ Model not found yet. Awaiting Colab download at: {model_path}")

class RecommendRequest(BaseModel):
    user_id: int
    movie_title: str
    top_n: int = 10

def get_sentiment_multiplier(text):
    if not isinstance(text, str) or text.strip() == "":
        return 1.0 
    
    try:
        # Colab trained model outputs LABEL_1 (Positive) and LABEL_0 (Negative)
        result = sentiment_analyzer(text[:512])[0] 
        if result['label'] == 'LABEL_1':
            return 1.0 + (result['score'] * 0.1)
        else:
            return 1.0 - (result['score'] * 0.1)
    except:
        return 1.0

@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    if req.movie_title not in indices:
        raise HTTPException(status_code=404, detail="Movie not found")
        
    idx = indices[req.movie_title]
    if isinstance(idx, pd.Series): 
        idx = idx.iloc[0]
        
    target_vector = tfidf_matrix[idx]
    sim_scores = cosine_similarity(target_vector, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-51:-1][::-1]
    sim_movies = metadata.iloc[top_indices].copy()
    
    sim_movies['svd_rating'] = sim_movies['movieId'].apply(
        lambda x: svd_model.predict(req.user_id, x).est
    )
    
    # --- APPLY SENTIMENT RE-RANKING ---
    sim_movies['sentiment_multiplier'] = sim_movies['overview'].apply(get_sentiment_multiplier)
    sim_movies['final_iq_score'] = sim_movies['svd_rating'] * sim_movies['sentiment_multiplier']
    
    top_results = sim_movies.sort_values('final_iq_score', ascending=False).head(req.top_n)
    
    return {"results": top_results[['original_title', 'svd_rating', 'sentiment_multiplier', 'final_iq_score', 'overview']].to_dict(orient="records")}