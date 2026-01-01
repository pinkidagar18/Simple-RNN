"""
IMDB Sentiment Analysis - Clean Dark Theme
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# Config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬", layout="wide")

# Clean Dark Theme CSS - No Boxes!
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
    
    /* Main background - solid dark */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: #0f0f23 !important;
        color: #ffffff;
    }
    
    /* Remove ALL default Streamlit containers/boxes */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        background: transparent !important;
    }
    
    div[data-testid="stVerticalBlock"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    div[data-testid="column"] {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem !important;
    }
    
    /* Title */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #f093fb 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -1px;
    }
    
    h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Glass cards - but only where we want them */
    .custom-card {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Text area - WHITE TEXT THAT YOU CAN SEE! */
    .stTextArea textarea {
        background: rgba(30, 30, 50, 0.8) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 16px !important;
        color: #ffffff !important;
        font-size: 1.1rem !important;
        padding: 1.2rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3) !important;
        background: rgba(40, 40, 60, 0.9) !important;
    }
    
    .stTextArea label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 1rem 2rem !important;
        border-radius: 16px !important;
        border: none !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Result cards */
    .result-card {
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        border: 2px solid;
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .positive-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(102, 126, 234, 0.15));
        border-color: #10b981;
    }
    
    .negative-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(240, 147, 251, 0.15));
        border-color: #ef4444;
    }
    
    .result-emoji {
        font-size: 5rem;
        margin-bottom: 1rem;
        animation: bounce 0.6s ease;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    
    .result-text {
        font-size: 2.5rem;
        font-weight: 900;
        margin: 0;
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(102, 126, 234, 0.3);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        transition: all 0.4s;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #f093fb);
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4);
        border-color: rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
    }
    
    .metric-value {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
        line-height: 1;
    }
    
    .metric-label {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Streamlit elements */
    .stMarkdown p {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 1.1rem;
    }
    
    /* Hide elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header {visibility: hidden;}
    
    /* Alert boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MAX_LEN = 500
MODEL_PATH = "simple_rnn_imdb.h5"

# Load model
@st.cache_resource
def load_resources():
    return load_model(MODEL_PATH), imdb.get_word_index()

with st.spinner('🎬 Loading AI Model...'):
    model, word_index = load_resources()

# Functions
def preprocess(text):
    if not text.strip():
        return None
    words = text.lower().split()
    encoded = [word_index.get(w, 2) + 3 for w in words]
    return pad_sequences([encoded], maxlen=MAX_LEN)

def predict(text):
    processed = preprocess(text)
    if processed is None:
        return None, None
    score = model.predict(processed, verbose=0)[0][0]
    return "Positive" if score >= 0.5 else "Negative", float(score)

def create_gauge(score, sentiment):
    color = "#10b981" if sentiment == "Positive" else "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "AI Confidence", 
            'font': {'size': 24, 'color': 'white', 'family': 'Inter', 'weight': 700}
        },
        number={
            'suffix': "%", 
            'font': {'size': 60, 'color': color, 'family': 'Inter', 'weight': 900}
        },
        gauge={
            'axis': {
                'range': [None, 100], 
                'tickwidth': 3, 
                'tickcolor': 'rgba(255,255,255,0.4)',
                'tickfont': {'size': 14, 'color': 'rgba(255,255,255,0.6)'}
            },
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(20, 20, 40, 0.6)",
            'borderwidth': 3,
            'bordercolor': "rgba(102, 126, 234, 0.3)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.15)'},
                {'range': [50, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
            ],
            'threshold': {
                'line': {'color': "#667eea", 'width': 5},
                'thickness': 0.8,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Inter"},
        height=320,
        margin=dict(l=10, r=10, t=70, b=10)
    )
    
    return fig

# Sample reviews
SAMPLES = {
    "🌟 Amazing": "This movie was absolutely incredible! The cinematography was breathtaking, the acting was superb, and the story kept me on the edge of my seat throughout.",
    "👎 Terrible": "What a complete waste of time and money. Poor acting, predictable plot, and terrible pacing. I want my money back. Not worth watching.",
    "🤔 Mixed": "The movie had stunning visuals and great music, but the story felt weak and the characters were underdeveloped. It's okay but not great.",
    "😍 Masterpiece": "One of the best films I've seen this year! Every scene was beautifully crafted. The emotional depth was extraordinary. Must watch!",
    "😤 Disappointing": "Boring, slow, and uninspired. The director clearly had no vision. I fell asleep twice. Save yourself the trouble and skip this one."
}

# Header
st.markdown('<h1>🎬 AI Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.3rem; color: rgba(255,255,255,0.7); margin-bottom: 3rem; font-weight: 500;">Discover the emotion behind movie reviews with Deep Learning</p>', unsafe_allow_html=True)

# Main layout
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    # Get sample if loaded
    default_text = st.session_state.get('sample_text', '')
    
    st.markdown("### ✍️ Enter Your Movie Review")
    review = st.text_area(
        "",
        value=default_text,
        placeholder="Type your honest movie review here... Be as detailed as you'd like!",
        height=200,
        label_visibility="collapsed"
    )
    
    # Clear sample
    if 'sample_text' in st.session_state:
        del st.session_state.sample_text
    
    # Stats
    if review:
        words = len(review.split())
        chars = len(review)
        st.markdown(f"<p style='color: rgba(255,255,255,0.6); font-size: 0.95rem; margin-top: 0.5rem;'>📝 {words} words  •  🔤 {chars} characters</p>", unsafe_allow_html=True)
    
    # Analyze button
    analyze = st.button("🚀 Analyze Sentiment", use_container_width=True)

with col_right:
    st.markdown("### 🎯 Quick Test Samples")
    
    for label, text in SAMPLES.items():
        if st.button(label, key=label, use_container_width=True):
            st.session_state.sample_text = text
            st.rerun()

# Results
if analyze:
    if not review.strip():
        st.warning("⚠️ Please enter a movie review to analyze")
    else:
        with st.spinner('🤖 AI is analyzing your review...'):
            sentiment, score = predict(review)
        
        if sentiment:
            # Result card
            card_class = "positive-card" if sentiment == "Positive" else "negative-card"
            emoji = "😊" if sentiment == "Positive" else "😞"
            
            st.markdown(f"""
                <div class="result-card {card_class}">
                    <div class="result-emoji">{emoji}</div>
                    <h2 class="result-text">{sentiment.upper()} SENTIMENT</h2>
                    <p style="font-size: 1.2rem; margin-top: 1rem; color: rgba(255,255,255,0.8); font-weight: 500;">
                        AI detected {sentiment.lower()} emotions in your review
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Metrics section with better styling
            st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
            
            metric_col1, metric_col2, metric_col3, gauge_col = st.columns([1, 1, 1, 2], gap="medium")
            
            with metric_col1:
                confidence = score * 100 if sentiment == "Positive" else (1 - score) * 100
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{confidence:.0f}%</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(review.split())}</div>
                        <div class="metric-label">Words</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{score:.2f}</div>
                        <div class="metric-label">Raw Score</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with gauge_col:
                # Gauge with custom container
                st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                        border: 1px solid rgba(102, 126, 234, 0.3);
                        border-radius: 20px;
                        padding: 1rem;
                        backdrop-filter: blur(20px);
                        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                    ">
                """, unsafe_allow_html=True)
                fig = create_gauge(score, sentiment)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.5);">
        <p style="font-size: 1rem; font-weight: 600; letter-spacing: 1px;">⚡ POWERED BY TENSORFLOW & DEEP LEARNING ⚡</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">Built with 25,000 IMDB Reviews • Simple RNN Architecture</p>
    </div>
""", unsafe_allow_html=True)