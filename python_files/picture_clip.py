import os
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm

# Configuration
IMAGE_DIR = "images"                    # Directory of cover images
IMAGE_OUTPUT = "picture_features.npy"   # Output for picture feature vectors
BVID_OUTPUT = "picture_bvids.npy"       # Ordered list of BVIDs (for tracking)

# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def extract_image_features(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model.encode_image(image_input)
        return feature.cpu().numpy().squeeze()
    except Exception as e:
        print(f"[ERROR] Failed to process image {image_path}: {e}")
        return None

def main():
    features = []
    bvids = []

    image_files = sorted(os.listdir(IMAGE_DIR))
    for file in tqdm(image_files, desc="Extracting CLIP features"):
        if not file.lower().endswith(".jpg"):
            continue
        bvid = os.path.splitext(file)[0]
        image_path = os.path.join(IMAGE_DIR, file)
        feature = extract_image_features(image_path)
        if feature is not None:
            features.append(feature)
            bvids.append(bvid)

    np.save(IMAGE_OUTPUT, np.array(features))
    np.save(BVID_OUTPUT, np.array(bvids))

    print(f"[DONE] Saved {len(features)} image vectors to {IMAGE_OUTPUT}")
    print(f"[INFO] Corresponding bvids saved to {BVID_OUTPUT}")

if __name__ == "__main__":
    main()
