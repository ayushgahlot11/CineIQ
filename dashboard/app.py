import streamlit as st
import pandas as pd
import requests
import visuals

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="CineIQ Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🎬 CineIQ: Hybrid Recommendation Engine")
st.markdown("An explainable, ensemble-based movie recommender.")

# --- 2. LOAD LIGHTWEIGHT DATASET FOR UI ---
@st.cache_resource
def load_ui_data():
    meta = pd.read_csv("../data/processed/master_metadata.csv").dropna(subset=['original_title'])
    ratings = pd.read_csv("../data/processed/ratings_sample.csv")
    return meta, ratings

metadata, ratings_df = load_ui_data()

# --- 3. THE UI SIDEBAR ---
st.sidebar.header("User Parameters")
user_id = st.sidebar.number_input("Target User ID (MovieLens)", min_value=1, max_value=200000, value=1)

# Grab unique movie titles for dropdown
movie_list = sorted(metadata['original_title'].unique().tolist())
target_movie = st.sidebar.selectbox("Select a Movie You Like", movie_list, index=movie_list.index("The Matrix") if "The Matrix" in movie_list else 0)

# --- 3.5 USER TASTE PROFILE ---
st.markdown(f"### 👤 Profile: User {user_id}")
col_chart, col_stats = st.columns([1, 1])

with col_chart:
    st.markdown("**Top Themes Radar**")
    radar_fig = visuals.generate_genre_radar(user_id, ratings_df, metadata)
    if radar_fig:
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.caption("Not enough data to build radar chart.")

with col_stats:
    st.markdown("**Quick Stats**")
    user_movie_count = len(ratings_df[ratings_df['userId'] == user_id])
    avg_rating = ratings_df[ratings_df['userId'] == user_id]['rating'].mean()
    
    st.metric("Total Movies Rated", user_movie_count)
    st.metric("Average Rating Given", f"{avg_rating:.2f} / 5.0" if pd.notna(avg_rating) else "N/A")

st.divider()

# --- 4. API ROUTING INTEGRATION ---
if st.sidebar.button("Generate Recommendations", type="primary"):
    st.subheader(f"Because you liked **{target_movie}**...")
    
    # Payload matching our FastAPI Pydantic schema
    payload = {
        "user_id": int(user_id),
        "movie_title": str(target_movie),
        "top_n": 10
    }
    
    API_URL = "http://127.0.0.1:8000/recommend"
    
    try:
        with st.spinner("Pinging FastAPI Backend & Running Custom DistilBERT Re-Ranker..."):
            response = requests.post(API_URL, json=payload)
            
        if response.status_code == 200:
            top_10 = response.json()["results"]
            
            # --- 5. DISPLAY RESULTS & EXPLAINABILITY ---
            for i, row in enumerate(top_10):
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        st.markdown(f"### 🏆 #{i + 1}")
                        st.metric(
                            label="Final IQ Score", 
                            value=f"{row['final_iq_score']:.2f}", 
                            delta=f"{row['sentiment_multiplier'] - 1.0:.3f} Sentiment"
                        )
                        
                    with col2:
                        st.markdown(f"#### {row['original_title']}")
                        st.caption(f"{row['overview']}")
                        
                        sentiment_str = "positive" if row['sentiment_multiplier'] > 1.0 else "mixed/negative"
                        st.info(f"💡 **Why this?** It shares strong thematic links with '{target_movie}'. Users similar to you rated it highly (SVD Matrix Prediction: {row['svd_rating']:.2f}), and your custom cloud-trained DistilBERT model evaluated its audience reception as {sentiment_str}.")
                    st.divider()
        else:
            st.error(f"🚨 Backend API error: {response.json().get('detail', 'Unknown error')}")
            
    except requests.exceptions.ConnectionError:
        st.error("🚨 Could not connect to the API. Make sure your FastAPI server is running on http://127.0.0.1:8000")