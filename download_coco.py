import os
import zipfile

import requests
from tqdm import tqdm


def download_and_unzip(url, target_dir):
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Get the filename from the URL
    filename = url.split("/")[-1]
    filepath = os.path.join(target_dir, filename)

    # Download the file
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(filepath, "wb") as file,
        tqdm(
            desc=filename,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar,
    ):
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

    # Unzip the file
    print(f"Unzipping {filename}...")
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    # Remove the zip file
    os.remove(filepath)
    print(f"Removed {filename}")


def main():
    target_dir = "./data/coco"
    urls = [
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
    ]

    for url in urls:
        download_and_unzip(url, target_dir)

    print("Download and unzip process completed.")


if __name__ == "__main__":
    main()
