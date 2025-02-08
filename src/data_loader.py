import os
import zipfile
import urllib.request
from io import BytesIO

def download_and_extract(url, extract_to):
    """Download and extract a ZIP file from a URL, saving CSVs directly ."""
    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                zip_data = BytesIO(response.read())
                with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                    for file in zip_ref.namelist():
                        if file.endswith('.csv'):
                            with zip_ref.open(file) as source, open(os.path.join(extract_to, os.path.basename(file)), 'wb') as target:
                                target.write(source.read())
                print(f"Extracted CSV files from {url} to {extract_to}")
            else:
                print(f"Failed to download {url}, status code: {response.status}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

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
