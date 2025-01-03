import cv2
import os
import time
from google.cloud import storage,firestore
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.ndimage import gaussian_filter

# Configuration variables
BUCKET_NAME = "vimarsh-a3197.appspot.com"  # Replace with your GCS bucket name
IMAGE_CAPTURE_DIR = "./captured_images"  # Directory to save captured images
SASNET_MODEL_PATH = "SASNet/models/SHHB.pth"  # Path to your SASNet model weights
RTMP_STREAM_URL = "rtmp://34.44.171.208/live"  # Replace with your RTMP stream URL
CAPTURE_INTERVAL = 2  # Capture interval in seconds
LOG_PARA = 1000
GAUSSIAN_SIGMA = 1.0
GAUSSIAN_KERNEL_SIZE = 5

db=firestore.Client()
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

def generate_metadata(path,count):
	metadata={ "gps_coordinates": "9.9252° N, 78.1198° E",
        "head_count":int(count),
        "id": "10473",
        "image_url":str(path),
        "pincode": "625001",
        "status": "active",
        "timestamps": "2024-09-05T09:40:00Z"}
	return metadata
def upload_metadata(metadata):
	try:
	   db.collection("HeadCount").add(metadata)
	   print("metadata successfully uploaded to firebase ")
	except Exception as e:
	   print(f"Error uploading metadata to firebase:{e}")

def main():
    # Ensure the image capture directory exists
    if not os.path.exists(IMAGE_CAPTURE_DIR):
        os.makedirs(IMAGE_CAPTURE_DIR)

    # Initialize video capture from RTMP stream
    cap = cv2.VideoCapture(RTMP_STREAM_URL)

    if not cap.isOpened():
        print("Error: Unable to open RTMP stream.")
        return

    frame_count = 0
    print("Starting to capture frames from RTMP stream...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from RTMP stream.")
            break

        # Save a frame every `CAPTURE_INTERVAL` seconds
        if frame_count % (CAPTURE_INTERVAL * int(cap.get(cv2.CAP_PROP_FPS))) == 0:
            timestamp = int(time.time())
            image_path = os.path.join(IMAGE_CAPTURE_DIR, f"frame_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Captured frame and saved to {image_path}")

            # Perform inference on the captured image
            print(f"Performing inference on {image_path}...")
            pred_count = perform_inference(image_path)
            print(f"Inference result: Predicted count = {pred_count:.2f}")
	   
	   #generate the metadata of the captured image
            metadata=generate_metadata(image_path,pred_count)
	   #upload the metadata of the captured image
            upload_metadata(metadata)
        frame_count += 1

    # Release the video capture object
    cap.release()
    print("RTMP stream closed.")

if __name__ == "__main__":
    main()
