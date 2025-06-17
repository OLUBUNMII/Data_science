# utils/kaggle_downloader.py

import os
import zipfile

def download_and_extract(dataset_slug, output_folder="data"):
    """
    Downloads and extracts a Kaggle dataset using the provided slug.
    
    Args:
        dataset_slug (str): The Kaggle dataset slug, e.g., "mlg-ulb/creditcardfraud"
        output_folder (str): Folder to extract the dataset into
    """
    #prevent re-downloading
    if os.path.exists(output_folder):
        print("Data already exists, skipping download.")
        return

    # Download the dataset zip
    print(f"Downloading dataset: {dataset_slug}")
    os.system(f'kaggle datasets download -d {dataset_slug}')

    # Build the expected zip filename
    filename = dataset_slug.split("/")[-1] + ".zip"

    # Unzip into target folder
    print(f"Extracting {filename} into {output_folder}/")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(output_folder) 
    print("Download and extraction complete.")

   
#To use in main.py
    
# from utils.kaggle_downloader import download_and_extract

# download_and_extract("mlg-ulb/creditcardfraud", output_folder="data")