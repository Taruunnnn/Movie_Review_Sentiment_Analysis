import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Load the model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# Set the title and description of the app
st.title('🎬 Movie Review Sentiment Analysis 🎥')
st.markdown('''
Enter a movie review to predict whether the sentiment is **positive** or **negative**. 
This tool uses a trained machine learning model to analyze the sentiment.
''')

# Display logo image
logo = Image.open('images/logo.jpg') 
st.image(logo, width=200)

# Background image styling
st.markdown(
    '''
    <style>
    body {
        background-size: cover;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# Input text box for movie review
st.subheader('📝 Enter Your Movie Review:')
review = st.text_area('Enter your movie review here:', height=150)

# Predict button
if st.button('Predict Sentiment'):
    if review.strip():
        review_scale = scaler.transform([review]).toarray()
        result = model.predict(review_scale)
        if result[0] == 0:
            st.markdown('<h3 style="color: red;">Negative Review</h3>', unsafe_allow_html=True)
        else:
            st.markdown('<h3 style="color: green;">Positive Review</h3>', unsafe_allow_html=True)
    else:
        st.write('Please enter a review before predicting.')

# Decorative image
decorative_image = Image.open('images/decorative_image.jpg')  
st.image(decorative_image, use_column_width=True)

# Add some additional styling
st.markdown(
    '''
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    </style>
    ''',
    unsafe_allow_html=True
)
