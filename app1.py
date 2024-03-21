# Import necessary libraries for Streamlit
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import time
import requests
from io import BytesIO

model_url = 'https://drive.google.com/file/d/1UOj5kZB2Nv5R_UNn9_09cundJH8qT0Wv/view?usp=sharing'

# Function to download the model file from Google Drive and save it locally
@st.cache_resource
def download_model(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as f:
        f.write(response.content)

# Download the model file from Google Drive
model_filename = 'my_CNN.h5'
download_model(model_url, model_filename)

# Load the trained model from the downloaded file
@st.cache_resource
def load_remote_model():
    model = load_model(model_filename)
    return model

model = load_remote_model()

# Load the trained model
#model = load_model('https://drive.google.com/file/d/1UOj5kZB2Nv5R_UNn9_09cundJH8qT0Wv/view?usp=sharing')

def preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.resize((150, 150))  # Resize image to match model's expected sizing
    if img.mode != 'RGB':
        img = img.convert('RGB')  # Convert image to RGB format if it's not already
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, [0])  # Add batch dimension
    return img

def main():
    st.title('Surface Crack Detection')
    st.markdown("Using Convolutional Neural Network (CNN)")
    uploaded_file = st.file_uploader('Upload an image of a concrete surface', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        img = preprocess_image(uploaded_file)
        with st.spinner('AI 🤖 is Working on it ...'):  # Display spinner while predicting
            time.sleep(2)  # Simulate a delay for demonstration purposes
            st.image(img[0], caption='Uploaded Image', use_column_width=True)
            prediction = model.predict(img)

        if prediction[0][0] >= 0.5:
            st.write("Surface type : Cracked Surface")
        else:
            st.write("Surface type : No Cracks Found")

    st.caption("Project completed by Sandeep")    

if __name__ == '__main__':
    main()

