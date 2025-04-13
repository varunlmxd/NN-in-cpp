import os
import pandas as pd
import numpy as np
import urllib.request
from zipfile import ZipFile

# Constants
URL = 'https://www.kaggle.com/api/v1/datasets/download/oddrationale/mnist-in-csv'
ZIP_FILE = 'mnist-in-csv.zip'
EXTRACT_FOLDER = 'mnist-in-csv'
OUTPUT_DIR = './'

# Download the dataset if not already downloaded
if not os.path.exists(ZIP_FILE):
    print(f"Downloading from Kaggle URL...\n{URL}")
    urllib.request.urlretrieve(URL, ZIP_FILE)

# Unzip the dataset
if not os.path.exists(EXTRACT_FOLDER):
    print("Extracting zip...")
    with ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)
    print("Extraction complete.")

# Load the CSVs
train_csv = os.path.join(EXTRACT_FOLDER, 'mnist_train.csv')
test_csv = os.path.join(EXTRACT_FOLDER, 'mnist_test.csv')

train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

# Split features and labels
X = train_data.drop(columns='label').values.astype('float32')
y = train_data['label'].values.astype('uint8')
X_test = test_data.drop(columns='label').values.astype('float32')
y_test = test_data['label'].values.astype('uint8')

# Clean up
del train_data, test_data

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
save_if_not_exists(os.path.join(OUTPUT_DIR, 'X_mnist_train.csv'), X, '%.16f')
save_if_not_exists(os.path.join(OUTPUT_DIR, 'y_mnist_train.csv'), y, '%d')
save_if_not_exists(os.path.join(OUTPUT_DIR, 'X_mnist_test.csv'), X_test, '%.16f')
save_if_not_exists(os.path.join(OUTPUT_DIR, 'y_mnist_test.csv'), y_test, '%d')

print("All done! CSV files are ready.")
