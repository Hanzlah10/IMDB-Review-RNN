import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import time

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Pro",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        /* Base styles */
        .main {
            padding: 2rem;
        }
        

        .stTextInput > div > div > input,
        .stSelectbox > div > div,
        .stMultiSelect > div > div {
            background-color: var(--background-color);
            color: var(--text-color);
            border-color: var(--secondary-background-color);
        }
        

        .streamlit-expanderHeader {
            background-color: var(--secondary-background-color) !important;
            color: var(--text-color) !important;
        }
        

        .sentiment-box {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 10px 0;
        }
        
        .positive {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .negative {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        

        [data-theme="light"] .positive {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        [data-theme="light"] .negative {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        

        [data-theme="dark"] .positive {
            background-color: #1e4620;
            color: #d4edda;
            border: 1px solid #2a6234;
        }
        [data-theme="dark"] .negative {
            background-color: #4c1c1f;
            color: #f8d7da;
            border: 1px solid #662427;
        }
        

        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
        

        .feedback-box {
            padding: 20px;
            border-radius: 5px;
            background-color: var(--secondary-background-color);
            margin: 10px 0;
        }
        
        /* Highlight box */
        .highlight {
            background-color: var(--secondary-background-color);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        
        /* Button hover state */
        .stButton>button:hover {
            border-color: var(--primary-color);
            color: var(--primary-color);
        }
        
        /* Metrics color */
        .css-1wivap2 {
            color: var(--text-color);
        }
        
        /* Info box dark mode compatibility */
        .stAlert {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
        }
    </style>
""", unsafe_allow_html=True)


# Constants
MAX_WORDS = 10000
MAX_LEN = 500

# Cache the model loading
@st.cache_resource
def load_my_model():
    try:
        model = load_model('imdb_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Cache the word index loading
@st.cache_resource
def get_word_index():
    try:
        word_index = imdb.get_word_index()
        word_index = {k:(v+3) for k,v in word_index.items()}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2
        word_index["<UNUSED>"] = 3
        return word_index
    except Exception as e:
        st.error(f"Error loading word index: {e}")
        return None

# Preprocess function
def preprocess_text(text):
    words = text.lower().split()
    indices = []
    for word in words:
        if word in word_index:
            idx = word_index[word]
            if idx < MAX_WORDS:
                indices.append(idx)
        else:
            indices.append(2)
    padded = sequence.pad_sequences([indices], maxlen=MAX_LEN)
    return padded

# Load model and word index
model = load_my_model()
word_index = get_word_index()

# Sidebar content
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/docs/main/public/logo.svg", width=200)
    st.title("✨ Pro Sentiment Analyzer")
    st.markdown("---")
    
    st.subheader("🎯 How it Works")
    st.write("""
    This advanced sentiment analyzer uses deep learning to understand the emotional 
    tone of movie reviews. It can detect subtle nuances in language to determine 
    whether a review is positive or negative.
    """)
    
    st.markdown("---")
    st.subheader("🎬 Pro Tips")
    with st.expander("Writing Effective Reviews"):
        st.write("""
        - Be specific about what you liked/disliked
        - Use descriptive language
        - Include details about acting, plot, effects
        - Compare to similar movies
        - Mention standout moments
        """)

# Main content
st.title("🎭 Professional Movie Review Analyzer")
st.markdown("### Unlock the Emotional Impact of Your Review")

# Create two columns for main content
col1, col2 = st.columns([2, 1])

with col1:
    # Review input section
    st.subheader("📝 Your Review")
    review_text = st.text_area(
        "",
        height=150,
        placeholder="Share your thoughts about the movie here...",
        key="review_input"
    )
    
    # Analysis button with custom styling
    analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)

    if analyze_button:
        if not review_text:
            st.warning("⚠️ Please enter a review first.")
        else:
            try:
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate analysis steps
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text("Preprocessing text...")
                    elif i < 60:
                        status_text.text("Analyzing sentiment...")
                    elif i < 90:
                        status_text.text("Generating insights...")
                    else:
                        status_text.text("Finalizing results...")
                    time.sleep(0.01)
                
                # Process the review
                processed_text = preprocess_text(review_text)
                prediction = model.predict(processed_text)
                score = float(prediction[0])
                
                # Clear status elements
                status_text.empty()
                progress_bar.empty()
                
                # Display results
                st.markdown("### 📊 Analysis Results")
                
                # Sentiment box
                sentiment = "Positive" if score > 0.5 else "Negative"
                confidence = score if score > 0.5 else 1 - score
                
                st.markdown(
                    f"""
                    <div class="sentiment-box {'positive' if sentiment == 'Positive' else 'negative'}">
                        <h2>{sentiment} Review</h2>
                        <h3>Confidence: {confidence:.1%}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Detailed metrics
                st.markdown("### 📈 Detailed Metrics")
                col_metrics1, col_metrics2 = st.columns(2)
                
                with col_metrics1:
                    st.metric("Word Count", len(review_text.split()))
                    st.metric("Sentiment Score", f"{score:.3f}")
                
                with col_metrics2:
                    st.metric("Processing Time", "0.5 seconds")
                    st.metric("Confidence Level", f"{confidence:.1%}")
                
                # Review feedback
                st.markdown("### 💡 Review Feedback")
                with st.expander("Click to see detailed feedback"):
                    st.markdown(
                        f"""
                        <div class="feedback-box">
                            <h4>Review Strength: {'Strong' if confidence > 0.8 else 'Moderate'}</h4>
                            <p>Your review shows a {'very clear' if confidence > 0.8 else 'moderate'} {sentiment.lower()} sentiment.</p>
                            <h4>Suggestions:</h4>
                            <ul>
                                <li>{'Consider adding more specific details' if len(review_text.split()) < 50 else 'Good level of detail provided'}</li>
                                <li>{'Try using more descriptive language' if confidence < 0.8 else 'Strong emotional content detected'}</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
            except Exception as e:
                st.error("🚫 An error occurred during analysis. Please try again.")
                st.exception(e)

with col2:
    # Example reviews section
    st.subheader("📚 Example Reviews")
    with st.expander("Positive Review Example"):
        st.markdown("""
        <div class="highlight">
        "An absolute masterpiece! The director's vision shines through in every scene, 
        with stunning cinematography and powerful performances from the entire cast. 
        The plot keeps you engaged from start to finish, and the score perfectly 
        complements the emotional journey."
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("Negative Review Example"):
        st.markdown("""
        <div class="highlight">
        "Unfortunately, this film falls short in every aspect. The plot is 
        predictable, the dialogue feels forced, and the special effects look 
        dated. Despite the talented cast, the poor direction and weak script 
        make this a disappointing experience."
        </div>
        """, unsafe_allow_html=True)
    
    # Tips section
    st.subheader("💡 Quick Tips")
    st.info("""
    - Be specific about what you liked or disliked
    - Mention technical aspects (directing, acting, effects)
    - Compare to similar movies
    - Discuss emotional impact
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🚀 Powered by Advanced AI • Made with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)