import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from collections import Counter

def generate_genre_radar(user_id, ratings_df, metadata_df):
    """
    Generates a Plotly Radar Chart showing a user's favorite genres/themes 
    based on movies they rated >= 3.5, filtering out common text noise.
    """
    # 1. Filter for movies the user actually liked
    user_history = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['rating'] >= 3.5)]
    
    # 2. Merge with metadata to get the features
    liked_movies = user_history.merge(metadata_df, on='movieId')
    
    if liked_movies.empty:
        return None
        
    # Define common structural stop words to screen out text noise
    stop_words = {
        'and', 'of', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 
        'by', 'from', 'is', 'it', 'that', 'this', 'as', 'are', 'was', 'about',
        'who', 'whom', 'his', 'her', 'their', 'he', 'she', 'they', 'man', 'woman',
        'young', 'out', 'up', 'new', 'one', 'two', 'find', 'finds', 'takes', 'place'
    }
    
    # 3. Count the themes
    theme_list = []
    for features in liked_movies['combined_features']:
        if pd.notna(features):
            # Clean punctuation, lowercase, and split into individual words
            words = [w.lower().strip(".,()?!:;\"'") for w in features.split()]
            # Keep only meaningful keywords (ignore stop words and tiny strings)
            filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
            theme_list.extend(filtered_words)
            
    theme_counts = Counter(theme_list)
    
    # Take the top 6 themes for a clean radar shape
    top_themes = dict(theme_counts.most_common(6))
    
    if not top_themes:
        return None

    # 4. Build the Plotly Radar Chart (Capitalize keys for visual cleanliness)
    df = pd.DataFrame(dict(
        r=list(top_themes.values()),
        theta=[k.capitalize() for k in top_themes.keys()]
    ))
    
    fig = px.line_polar(df, r='r', theta='theta', line_close=True, 
                        color_discrete_sequence=['#00E676'])
    
    fig.update_traces(fill='toself', fillcolor='rgba(0, 230, 118, 0.2)')
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=20),
        height=300
    )
    return fig