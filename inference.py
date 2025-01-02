import os
from google.cloud import storage
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter

# Configuration variables
BUCKET_NAME = "vimarsh-a3197.appspot.com"  # Replace with your GCS bucket name
DOWNLOAD_DIR = "./images"  # Directory to save downloaded images
SASNET_MODEL_PATH = "models/SHHA.pth"  # Path to your SASNet model weights
LOG_PARA = 1000
GAUSSIAN_SIGMA = 1.0
GAUSSIAN_KERNEL_SIZE = 5

# Initialize SASNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Args:
    def __init__(self, block_size=32):
        self.block_size = block_size

# Initialize SASNet with args object
args = Args(block_size=32)
from model import SASNet  # Assuming your SASNet model is defined in `model.py`
sasnet_model = SASNet(args=args)
sasnet_model.load_state_dict(torch.load(SASNET_MODEL_PATH, map_location=device))
sasnet_model.to(device).eval()

def preprocess_image(image_path):
    """Preprocess the image for SASNet."""
    transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
    ])
    pil_image = Image.open(image_path).convert("RGB")
    return transform(pil_image).unsqueeze(0).to(device)

def apply_gaussian_filter(pred_map, sigma, kernel_size):
    """Apply Gaussian filter to the density map."""
    return gaussian_filter(pred_map, sigma=sigma, order=0)

def perform_inference(image_path):
    """Perform inference on a single image."""
    preprocessed_image = preprocess_image(image_path)
    with torch.no_grad():
        pred_map = sasnet_model(preprocessed_image).squeeze(0).cpu().numpy()
    pred_map_smoothed = apply_gaussian_filter(pred_map, GAUSSIAN_SIGMA, GAUSSIAN_KERNEL_SIZE)
    pred_count = np.sum(pred_map_smoothed) / LOG_PARA
    return pred_count

def download_new_images(bucket, processed_files):
    """Checks for new images in the bucket and downloads them."""
    blobs = bucket.list_blobs()
    new_files = []

    for blob in blobs:
        if blob.name not in processed_files:
            local_path = os.path.join(DOWNLOAD_DIR, blob.name)
            blob.download_to_filename(local_path)
            print(f"Downloaded {blob.name} to {local_path}")
            new_files.append(local_path)
            processed_files.add(blob.name)

    return new_files

def main():
    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # Ensure download directory exists
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    processed_files = set()

    print("Listening for new images in the bucket...")
    while True:
        new_files = download_new_images(bucket, processed_files)

        if new_files:
            for image_path in new_files:
                print(f"Performing inference on {image_path}...")
                pred_count = perform_inference(image_path)
                print(f"Inference result for {image_path}: Predicted count = {pred_count:.2f}")
        else:
            print("No new images found.")
        
        time.sleep(5)  # Polling interval

if __name__ == "__main__":
    main()
