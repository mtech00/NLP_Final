import os
import requests
import zipfile
from io import BytesIO

def download_and_extract(url, extract_to):
    """Download and extract a ZIP file from a URL, saving CSVs directly to extract_to."""
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith('.csv'):
                    with zip_ref.open(file) as source, open(os.path.join(extract_to, os.path.basename(file)), 'wb') as target:
                        target.write(source.read())
        print(f"Extracted CSV files from {url} to {extract_to}")
    else:
        print(f"Failed to download {url}, status code: {response.status_code}")

if __name__ == "__main__":
    # Define URLs
    urls = [
        "https://github.com/mtech00/EPAM_DS/raw/main/final_project_train_dataset.zip",
        "https://github.com/mtech00/EPAM_DS/raw/main/final_project_test_dataset.zip"
    ]
    
    # Define extraction directory
    extract_dir = "data/raw"
    os.makedirs(extract_dir, exist_ok=True)
    
    # Download and extract files
    for url in urls:
        download_and_extract(url, extract_dir)