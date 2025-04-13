import os
import urllib.request
from zipfile import ZipFile
import numpy as np
import cv2
from tqdm import tqdm

# Constants
URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'
OUTPUT_DIR = './'

# Download the dataset if not already downloaded
if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE}...')
    urllib.request.urlretrieve(URL, FILE)

# Unzip the dataset
print('Unzipping images...')
with ZipFile(FILE, 'r') as zip_ref:
    zip_ref.extractall(FOLDER)
print('Unzipping complete.')

# Function to load dataset
def load_mnist_fashion_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    X, y = [], []

    for label in tqdm(labels, desc=f"Loading {dataset}"):
        folder_path = os.path.join(path, dataset, label)
        for file in tqdm(os.listdir(folder_path), desc=f"Reading {label} images"):
            img_path = os.path.join(folder_path, file)
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')

# Function to create complete MNIST dataset
def create_data_mnist_fashion(path):
    X_train, y_train = load_mnist_fashion_dataset('train', path)
    X_test, y_test = load_mnist_fashion_dataset('test', path)
    return X_train, y_train, X_test, y_test

# Load and process data
X, y, X_test, y_test = create_data_mnist_fashion(FOLDER)

# Shuffle the training data
shuffle_idx = np.random.permutation(len(X))
X, y = X[shuffle_idx], y[shuffle_idx]

# Normalize and flatten data
X = ((X.astype(np.float32) - 127.5) / 127.5).reshape(-1, 28 * 28)
X_test = ((X_test.astype(np.float32) - 127.5) / 127.5).reshape(-1, 28 * 28)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper function to save if file doesn't exist
def save_if_not_exists(filepath, data, fmt):
    if not os.path.exists(filepath):
        print(f"Saving {filepath}...")
        np.savetxt(filepath, data, delimiter=',', fmt=fmt)
    else:
        print(f"{filepath} already exists, skipping.")

# Save data conditionally
save_if_not_exists(os.path.join(OUTPUT_DIR, 'X_fashion_mnist_train.csv'), X, '%.16f')
save_if_not_exists(os.path.join(OUTPUT_DIR, 'y_fashion_mnist_train.csv'), y, '%d')
save_if_not_exists(os.path.join(OUTPUT_DIR, 'X_fashion_mnist_test.csv'), X_test, '%.16f')
save_if_not_exists(os.path.join(OUTPUT_DIR, 'y_fashion_mnist_test.csv'), y_test, '%d')

print("All done! CSV files are ready.")