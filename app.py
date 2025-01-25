import streamlit as st
import pandas as pd
import requests
from transformers import pipeline
import io
import os

@st.cache_resource
def load_model():
    """
    Load the multilingual sentiment analysis model from Hugging Face.
    """
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

sentiment_pipeline = load_model()



@st.cache_data
def load_data():
    """
    Load the dataset from Google Drive using the file ID stored as an environment variable.
    """
    try:
        file_id = os.getenv("GOOGLE_DRIVE_FILE_ID")
        url = f"https://drive.google.com/uc?id={file_id}"
        response = requests.get(url)
        response.raise_for_status()

        return pd.read_csv(io.StringIO(response.text), parse_dates=['data'], encoding='utf-8')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

#############################
# SENTIMENT ANALYSIS
#############################
def analyze_sentiment(text):
    """
    Perform sentiment analysis and map star ratings to GOOD, BAD, or NEUTRAL.
    """
    try:
        result = sentiment_pipeline(text)
        label = result[0]['label']
        if "1 star" in label or "2 stars" in label:
            return "🔴 RUIM"
        elif "3 stars" in label:
            return "🟡 NEUTRO"
        else:
            return "🟢 BOM"
    except Exception as e:
        st.warning(f"Error analyzing sentiment for text: {text} -> {e}")
        return "NEUTRAL"

#############################
# STREAMLIT APP
#############################
st.title("📊 Análise dos feedbacks sobre influencers brasileiros - Em construção")

if not df.empty:
    profile_counts = df['perfil'].value_counts()
    frequent_profiles = profile_counts[profile_counts > 1].index.tolist()

    selected_profile = st.selectbox(
        "Clique ou escreva. Apenas perfis com mais de um feedback são exibidos.",
        options=frequent_profiles,
        help="Type to search for profiles with more than one entry."
    )

    if selected_profile:
        profile_data = df[df['perfil'] == selected_profile].copy()

        # Preprocess 'experiencia'
        profile_data['experiencia'] = profile_data['experiencia'].fillna("No content").astype(str)

        # Perform sentiment analysis if not done yet
        if 'sentiment_experiencia' not in profile_data.columns:
            #st.info("Analisando os sentimentos dos feedbacks...")
            profile_data['sentiment_experiencia'] = profile_data['experiencia'].apply(analyze_sentiment)

        # Calculate final sentiment
        sentiment_summary = profile_data['sentiment_experiencia'].value_counts().idxmax()
        average_rating = profile_data['nota'].mean()
        total_entries = len(profile_data)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Sentimento")
            st.markdown(f"## {sentiment_summary}")
        with col2:
            st.subheader("Média das notas")
            st.markdown(f"## {average_rating:.2f}")
        with col3:
            st.subheader("Nº de feedbacks")
            st.markdown(f"## {total_entries}")

    st.subheader("🏆 Top Influencers por média de notas")
    ranking_df = (
        df[df['perfil'].isin(frequent_profiles)]
        .groupby('perfil')
        .agg(avg_nota=('nota', 'mean'), total_entries=('nota', 'size'))
        .reset_index()
        .sort_values(by='avg_nota', ascending=False)
    )
    st.dataframe(ranking_df, use_container_width=True)
else:
    st.warning("No data available to display. Please check your dataset.")

st.markdown("---")
st.markdown("Developed by zLobo")