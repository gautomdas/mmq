import os
import requests
import zipfile
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

    with open(filepath, "wb") as file, tqdm(
        desc=filename,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
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
    target_dir = "./data/vqa2"
    images = "http://images.cocodataset.org/zips/val2017.zip"
    annotations = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
    questions = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"

    download_and_unzip(images, target_dir)
    download_and_unzip(annotations, target_dir+"/annotations")
    download_and_unzip(questions, target_dir+"/questions")

    print("Download and unzip process completed.")


if __name__ == "__main__":
    main()
