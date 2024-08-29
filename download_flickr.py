import os
import requests
import zipfile
from tqdm import tqdm


def download_and_unzip(url, target_dir):
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Get the filename from the URL
    filename = url.split("/")[-1].split("?")[0]
    filepath = os.path.join(target_dir, filename)
    file_extension = filename.split(".")[-1]

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
    if file_extension == "zip":
        print(f"Unzipping {filename}...")
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(target_dir)

        # Remove the zip file
        os.remove(filepath)
        print(f"Removed {filename}")


def main():
    target_dir = "./data/flickr30k"
    images_url = "https://huggingface.co/datasets/nlphuji/flickr_1k_test_image_text_retrieval/resolve/main/images_flickr_1k_test.zip?download=true"
    annotations_url = "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json"


    # Download annotations
    download_and_unzip(annotations_url, target_dir+"/annotations")

    # Download images
    download_and_unzip(images_url, target_dir)

    print("Download and unzip process completed.")


if __name__ == "__main__":
    main()
