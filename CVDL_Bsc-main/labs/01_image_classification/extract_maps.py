import torch
import torchvision.transforms.v2 as transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from model import CNN_classifier
import os

# ---------------------------------------------------------------------------------
# Load Trained Model
# ---------------------------------------------------------------------------------
MODEL_PATH = "model_cnn.pth"  # Ensure this is the correct trained model file
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model
model = CNN_classifier(n_classes=37).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()  # Set to evaluation mode

# ---------------------------------------------------------------------------------
# Feature Extraction Hook
# ---------------------------------------------------------------------------------
feature_maps = {}

def hook_fn(module, input, output):
    """ Hook function to store feature maps """
    feature_maps["conv_output"] = output

# Attach hook to the first convolutional layer
model.conv1.register_forward_hook(hook_fn)

# ---------------------------------------------------------------------------------
# Load a Sample Image
# ---------------------------------------------------------------------------------
transform_img = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize((430, 380), antialias=True),  # Ensure size matches model input
    transforms.ToDtype(torch.float32, scale=True),
])

def transform(samples):
    samples["img"] = [transform_img(img) for img in samples["img"]]
    return samples

# Load dataset and pick one image
ds = load_dataset("cvdl/oxford-pets")
ds = ds["valid"].select_columns(["img"]).with_transform(transform)
sample_img = ds[0]["img"].unsqueeze(0).to(DEVICE)  # Add batch dimension

# ---------------------------------------------------------------------------------
# Forward Pass & Extract Features
# ---------------------------------------------------------------------------------
with torch.no_grad():
    _ = model(sample_img)  # Run a forward pass to capture feature maps

# Get feature maps from the hook
conv1_output = feature_maps["conv_output"].cpu().numpy()
num_filters = conv1_output.shape[1]

# ---------------------------------------------------------------------------------
# Visualizing Feature Maps
# ---------------------------------------------------------------------------------
def visualize_feature_maps(feature_maps, save_dir="feature_maps"):
    """ Saves feature maps as images """
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_filters):
        plt.imshow(feature_maps[0, i], cmap="viridis")
        plt.axis("off")
        plt.savefig(f"{save_dir}/feature_map_{i}.png", bbox_inches="tight")
        plt.close()

    print(f"Feature maps saved in '{save_dir}/' directory.")

# Generate feature map images   
visualize_feature_maps(conv1_output)
