# quantize_blip2.py

import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import json
import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

# Suppress specific warnings if desired
warnings.filterwarnings("ignore", category=FutureWarning)

# =======================
# Configuration Parameters
# =======================

# Device configuration: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to COCO dataset
COCO_DIR = "./data/coco"  # Update this path to your COCO dataset directory
IMAGE_DIR = os.path.join(COCO_DIR, "val2017")
ANNOTATION_FILE = os.path.join(COCO_DIR, "annotations", "captions_val2017.json")

# Paths for saving models
LANGUAGE_MODEL_PATH = "./blip2_language_model"
QUANTIZED_MODEL_DIR = "./quantized_blip2_language_model"

# Number of samples for calibration and evaluation
CALIBRATION_SAMPLES = 128
EVALUATION_SAMPLES = 10

# =======================
# Dataset Definition
# =======================


class COCOSampleDataset(Dataset):
    """
    Custom Dataset for loading COCO captions and corresponding images.
    """

    def __init__(self, annotation_file, image_dir, num_samples=None):
        """
        Initializes the dataset by loading annotations.

        Args:
            annotation_file (str): Path to the COCO annotations JSON file.
            image_dir (str): Directory containing COCO validation images.
            num_samples (int, optional): Number of samples to load. Loads all if None.
        """
        with open(annotation_file, "r") as f:
            annotations = json.load(f)

        self.image_dir = image_dir
        self.samples = []
        image_id_to_file = {
            img["id"]: img["file_name"] for img in annotations["images"]
        }

        for ann in annotations["annotations"]:
            image_file = image_id_to_file.get(ann["image_id"], None)
            if image_file:
                self.samples.append(
                    {
                        "image_file": os.path.join(image_dir, image_file),
                        "caption": ann["caption"],
                    }
                )

        if num_samples:
            self.samples = random.sample(
                self.samples, min(num_samples, len(self.samples))
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =======================
# Main Quantization Script
# =======================


def main():
    print("Loading BLIP-2 model and processor...")
    model_name = "Salesforce/blip2-opt-2.7b"
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    model = model.to(device)
    model.eval()
    print("BLIP-2 model and processor loaded successfully.\n")

    if False:
        # -----------------------
        # Step 1: Load BLIP-2 Model and Processor
        # -----------------------
        print("Loading BLIP-2 model and processor...")
        model_name = "Salesforce/blip2-opt-2.7b"
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        model = model.to(device)
        model.eval()
        print("BLIP-2 model and processor loaded successfully.\n")

        # -----------------------
        # Step 2: Prepare Calibration Data
        # -----------------------
        print(
            f"Preparing {CALIBRATION_SAMPLES} calibration samples from {ANNOTATION_FILE}..."
        )
        calibration_dataset = COCOSampleDataset(
            annotation_file=ANNOTATION_FILE,
            image_dir=IMAGE_DIR,
            num_samples=CALIBRATION_SAMPLES,
        )

        calibration_data = []
        for sample in tqdm(calibration_dataset, desc="Processing calibration samples"):
            try:
                image = Image.open(sample["image_file"]).convert("RGB")
                inputs = processor(
                    images=image,
                    text=sample["caption"],
                    return_tensors="pt",
                    padding=True,
                )
                calibration_data.append(
                    {
                        "input_ids": inputs.input_ids.squeeze(0).to(device),
                        "attention_mask": inputs.attention_mask.squeeze(0).to(device),
                    }
                )
            except Exception as e:
                print(f"Error processing {sample['image_file']}: {e}")

        print(f"Prepared {len(calibration_data)} calibration samples.\n")

        # -----------------------
        # Step 3: Extract and Save the Language Model
        # -----------------------
        print("Extracting and saving the language model component...")
        language_model = model.language_model
        if not os.path.exists(LANGUAGE_MODEL_PATH):
            os.makedirs(LANGUAGE_MODEL_PATH)
        language_model.save_pretrained(LANGUAGE_MODEL_PATH)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(LANGUAGE_MODEL_PATH)
        print(f"Language model and tokenizer saved to {LANGUAGE_MODEL_PATH}.\n")

        # -----------------------
        # Step 4: Define Quantization Configuration
        # -----------------------
        print("Defining quantization configuration for AutoGPTQ...")
        quantize_config = BaseQuantizeConfig(
            bits=4,  # Number of bits for quantization
            group_size=128,  # Size of the groups for quantization
            desc_act=False,  # Disable activation description
        )
        print("Quantization configuration defined.\n")

        # -----------------------
        # Step 5: Quantize the Language Model Using AutoGPTQ
        # -----------------------
        print("Quantizing the language model using AutoGPTQ...")
        quantized_model = AutoGPTQForCausalLM.from_pretrained(
            LANGUAGE_MODEL_PATH,
            quantize_config=quantize_config,
            use_safetensors=True,  # Use safe tensors format
            trust_remote_code=True,  # Trust the model's remote code (set to False if unsure)
            device_map="auto",  # Automatically map model to available device
        )

        # Perform quantization with calibration data
        print("Starting quantization process...")
        quantized_model.quantize(calibration_data)
        print("Quantization completed.")

        # Save the quantized language model
        if not os.path.exists(QUANTIZED_MODEL_DIR):
            os.makedirs(QUANTIZED_MODEL_DIR)
        quantized_model.save_pretrained(QUANTIZED_MODEL_DIR)
        print(f"Quantized language model saved to {QUANTIZED_MODEL_DIR}.\n")

    # -----------------------
    # Step 6: Load the Quantized Language Model
    # -----------------------
    quantize_config = BaseQuantizeConfig(
        bits=4,  # Number of bits for quantization
        group_size=128,  # Size of the groups for quantization
        desc_act=False,  # Disable activation description
    )
    print("Loading the quantized language model...")

    quantized_language_model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path=QUANTIZED_MODEL_DIR,  # This specifies the local path
        quantize_config=quantize_config,  # Pass the quantization config
        device_map="auto",  # Automatically map to available device
        model_basename="gptq_model-4bit-128g.safetensors",  # Specify the correct model file
        use_safetensors=True,  # Ensure we're using safetensors
    )

    print("Quantized language model loaded successfully.\n")

    # -----------------------
    # Step 7: Replace the Language Model in BLIP-2 with the Quantized Version
    # -----------------------
    print(
        "Replacing the original language model in BLIP-2 with the quantized version..."
    )
    model.language_model = quantized_language_model
    model = model.to(device)
    model.eval()
    print("Language model replaced successfully.\n")

    # -----------------------
    # Step 8: Run Evaluation on the Quantized Model
    # -----------------------
    print(
        f"Running evaluation on {EVALUATION_SAMPLES} samples using the quantized model...\n"
    )
    evaluation_dataset = COCOSampleDataset(
        annotation_file=ANNOTATION_FILE,
        image_dir=IMAGE_DIR,
        num_samples=EVALUATION_SAMPLES,
    )

    for idx, sample in enumerate(evaluation_dataset, 1):
        try:
            image_path = sample["image_file"]
            original_caption = sample["caption"]

            # Process the image
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            # Generate caption using the quantized model
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=50)
            generated_caption = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Display results
            print(f"Sample {idx}:")
            print(f"Image: {os.path.basename(image_path)}")
            print(f"Original Caption: {original_caption}")
            print(f"Generated Caption: {generated_caption}\n")

        except Exception as e:
            print(f"Error during evaluation of {sample['image_file']}: {e}\n")

    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()

