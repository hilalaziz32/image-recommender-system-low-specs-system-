import pickle
import numpy as np
import tensorflow as tf
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import streamlit as st

# Initialize ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Add GlobalMaxPooling2D for feature extraction
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Load embeddings and filenames
feature_list = np.array(pickle.load(open("embeddings_merged.pkl", "rb")))
filenames = pickle.load(open("filenames_merged.pkl", "rb"))

# Streamlit app setup
st.title("Image Recommendation System")
st.write("Upload an image to find similar images")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the uploaded image
    img_array = np.array(image.resize((224, 224), Image.Resampling.LANCZOS))
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    # Extract features and normalize
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    # Fit NearestNeighbors and find similar images
    neighbours = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
    neighbours.fit(feature_list)

    distances, indices = neighbours.kneighbors([normalized_result])

    # Display the top 5 recommended images
    st.write("Top 5 Recommended Images:")
    for file in indices[0][1:6]:
        recommended_image = Image.open(filenames[file]).resize((60, 80), Image.Resampling.LANCZOS)
        st.image(recommended_image, caption=filenames[file], use_column_width=True)
