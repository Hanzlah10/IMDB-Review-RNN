import numpy as np
# import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #FF4B4B;
        color: white;
    }
    .sentiment-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive {
        background-color: #90EE90;
        color: #1E4620;
    }
    .negative {
        background-color: #FFB6C1;
        color: #8B0000;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
MAX_VOCAB_SIZE = 10000  # Match this with your model's vocabulary size
MAX_SEQUENCE_LENGTH = 500  # Match this with your model's sequence length

# Load the IMDB dataset word index
@st.cache_resource
def load_word_index():
    word_index = imdb.get_word_index()
    # Filter word_index to only include the most frequent words
    sorted_words = sorted(word_index.items(), key=lambda x: x[1])
    filtered_words = sorted_words[:MAX_VOCAB_SIZE]
    filtered_word_index = {word: (i + 3) for i, (word, _) in enumerate(filtered_words)}
    # Add special tokens
    filtered_word_index['<PAD>'] = 0
    filtered_word_index['<START>'] = 1
    filtered_word_index['<UNK>'] = 2
    filtered_word_index['<UNUSED>'] = 3
    reverse_word_index = {value: key for key, value in filtered_word_index.items()}
    return filtered_word_index, reverse_word_index

word_index, reverse_word_index = load_word_index()

# Load the pre-trained model
@st.cache_resource
def load_sentiment_model():
    return load_model('imdb_model.h5')

model = load_sentiment_model()

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review])

def preprocess_text(text):
    # Convert to lowercase and split
    words = text.lower().split()
    
    # Convert words to indices, using 2 (UNK) for words not in vocabulary
    encoded_review = []
    for word in words:
        # Get index from word_index, default to UNK (2) if word not found
        idx = word_index.get(word, 2)
        # Ensure index is within vocabulary size
        if idx >= MAX_VOCAB_SIZE:
            idx = 2  # UNK token
        encoded_review.append(idx)
    
    # Pad sequence
    padded_review = sequence.pad_sequences([encoded_review], maxlen=MAX_SEQUENCE_LENGTH)
    return padded_review

def create_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment Score", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "#FFB6C1"},
                {'range': [33, 66], 'color': "#FFE4B5"},
                {'range': [66, 100], 'color': "#90EE90"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# Main app layout
st.title('🎬 Movie Review Sentiment Analyzer')

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.write("""
    This app analyzes the sentiment of movie reviews using a deep learning model 
    trained on the IMDB dataset. Enter your review to see if it's predicted to be 
    positive or negative.
    """)
    st.markdown("---")
    st.subheader("Tips for best results:")
    st.write("• Write at least a few sentences")
    st.write("• Be specific about what you liked/disliked")
    st.write("• Use descriptive language")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter your movie review")
    user_input = st.text_area(
        "",
        height=200,
        placeholder="Type your movie review here..."
    )

    if st.button('Analyze Sentiment', key='analyze'):
        if user_input:
            try:
                with st.spinner('Analyzing your review...'):
                    # Preprocess and predict
                    preprocessed_input = preprocess_text(user_input)
                    prediction = model.predict(preprocessed_input)
                    score = prediction[0][0]
                    sentiment = 'Positive' if score > 0.5 else 'Negative'

                    # Display results
                    st.markdown("### Analysis Results")
                    
                    # Sentiment box
                    sentiment_class = "positive" if sentiment == "Positive" else "negative"
                    st.markdown(f"""
                        <div class="sentiment-box {sentiment_class}">
                            <h2 style="text-align: center; margin: 0;">
                                {sentiment} Review
                            </h2>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                    # Confidence gauge
                    st.plotly_chart(create_gauge_chart(score), use_container_width=True)

                    # Word count and analysis details
                    st.markdown("### Review Statistics")
                    word_count = len(user_input.split())
                    st.write(f"Word count: {word_count}")
                    st.write(f"Confidence score: {score:.2%}")

            except Exception as e:
                st.error("An error occurred during analysis. Please try again with a different review.")
                st.exception(e)
        else:
            st.error("Please enter a review before analyzing.")

with col2:
    st.subheader("Example Reviews")
    st.markdown("""
    **Positive Example:**
    *"This movie was absolutely brilliant! The acting was superb, and the plot kept me engaged throughout. The cinematography was breathtaking, and the score perfectly complemented each scene."*
    
    **Negative Example:**
    *"I was really disappointed with this film. The plot had numerous holes, the dialogue felt forced, and the special effects were dated. I wouldn't recommend it."*
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Streamlit • Powered by TensorFlow</p>
    </div>
    """,
    unsafe_allow_html=True
)