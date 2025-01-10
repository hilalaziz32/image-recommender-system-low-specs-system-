# Image Feature Extraction with ResNet50

This project demonstrates how to extract features from a set of images using the ResNet50 model from TensorFlow. The features are normalized and saved for later use in applications like image clustering, retrieval, or similarity detection.

## Overview
- **Feature Extraction:** Uses ResNet50 (pre-trained on ImageNet) without the top layer, combined with a `GlobalMaxPooling2D` layer.
- **Batch Processing:** Processes images in batches to handle large datasets and cater to systems with limited resources.
- **Output:** Extracted features are saved as `.pkl` files along with the corresponding filenames for each batch.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install numpy tensorflow tqdm
```

## Instructions

### 1. Prepare Your Images
Place all the images you want to process in a folder named `images` in the same directory as the script.

### 2. Run the Script
You have two options:

#### Option A: Process All Batches At Once
If your system can handle the entire dataset in one go, uncomment the entire script and run it. This will process all batches sequentially and save the results.

#### Option B: Process Batches One by One
For systems with limited resources:
1. Uncomment only one batch section (e.g., `Batch 1`) at a time.
2. Run the script for that batch.
3. Repeat the process for each batch.

### 3. Output Files
For each batch, the following files will be saved:
- `embeddings_batch_X.pkl`: Contains the extracted features.
- `filenames_batch_X.pkl`: Contains the filenames corresponding to the features.

### Example: Processing Batch 1
1. Uncomment the section for `Batch 1`.
2. Run the script:
   ```bash
   python feature_extraction.py
   ```
3. The following files will be generated:
   - `embeddings_batch_1.pkl`
   - `filenames_batch_1.pkl`

### Notes
- You can adjust the batch size by modifying the script. The current implementation splits the dataset into 5 equal parts.
- Ensure the `images` folder exists and contains your dataset before running the script.

### Troubleshooting
- **Memory Issues:** Use smaller batch sizes or process one batch at a time.
- **File Not Found:** Check if the `images` folder is correctly named and located in the script's directory.

## License
This project is open-source and available for personal or academic use.

## Dataset Link:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
