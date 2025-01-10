import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import os
from tqdm import tqdm

# Initialize ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Add GlobalMaxPooling2D for feature extraction
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to extract features
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Collect image filenames
filenames = []
for file in os.listdir("images"):
    filenames.append(os.path.join('images', file))

# Extract features and store them in a list
feature_list = []
# Step 1: Split filenames into three batches
batch_size = len(filenames) // 5
batches = [filenames[i:i + batch_size] for i in range(0, len(filenames), batch_size)]

################################################################
# # Process Batch 1
# batch_1 = batches[0]
#
# print("Processing Batch 1...")
# for file in tqdm(batch_1):
#     feature_list.append(extract_features(file, model))
#
# # Save Batch 1 results
# with open('embeddings_batch_1.pkl', 'wb') as f:
#     pickle.dump(feature_list, f)
#
# with open('filenames_batch_1.pkl', 'wb') as f:
#     pickle.dump(batch_1, f)
#
# print("Batch 1 saved. You can now close your laptop.")

##################################################################
# batch_2 = batches[1]
#
# print("Processing Batch 2...")
# for file in tqdm(batch_2):
#     feature_list.append(extract_features(file, model))
#
# # Save Batch 2 results
# with open('embeddings_batch_2.pkl', 'wb') as f:
#     pickle.dump(feature_list, f)
#
# with open('filenames_batch_2.pkl', 'wb') as f:
#     pickle.dump(batch_2, f)
#
# print("Batch 2 saved. You can now close your laptop.")

####################################################################
# batch_3 = batches[2]
#
# print("Processing Batch 3...")
# for file in tqdm(batch_3):
#     feature_list.append(extract_features(file, model))
#
# # Save Batch 3 results
# with open('embeddings_batch_3.pkl', 'wb') as f:
#     pickle.dump(feature_list, f)
#
# with open('filenames_batch_3.pkl', 'wb') as f:
#     pickle.dump(batch_3, f)
#
# print("Batch 3 saved. You can now close your laptop.")

##################################################################
# batch_4 = batches[3]
#
# print("Processing Batch 4...")
# for file in tqdm(batch_4):
#     feature_list.append(extract_features(file, model))
#
# # Save Batch 1 results
# with open('embeddings_batch_4.pkl', 'wb') as f:
#     pickle.dump(feature_list, f)
#
# with open('filenames_batch_4.pkl', 'wb') as f:
#     pickle.dump(batch_4, f)
#
# print("Batch 4 saved. You can now close your laptop.")


##################################################
# batch_5 = batches[4]
#
# print("Processing Batch 5...")
# for file in tqdm(batch_5):
#     feature_list.append(extract_features(file, model))
#
# # Save Batch 1 results
# with open('embeddings_batch_5.pkl', 'wb') as f:
#     pickle.dump(feature_list, f)
#
# with open('filenames_batch_5.pkl', 'wb') as f:
#     pickle.dump(batch_5, f)
#
# print("Batch 5 saved. You can now close your laptop.")
