import pandas as pd
import os
import zipfile
import requests

def download_and_load_kaggle_dataset(output_csv="sudoku.csv"):
    zip_path = "sudoku-puzzles.zip"
    url = "https://www.kaggle.com/api/v1/datasets/download/radcliffe/3-million-sudoku-puzzles-with-ratings"

    # Skip download if file already exists
    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        if response.status_code == 200:
            with open(zip_path, "wb") as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download: {response.status_code} - {response.text}")

    print("Reading dataset...")
    K = pd.read_csv(zip_path, index_col="id")
    return K

