import torch
import torchvision.transforms.v2 as transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
from model import CNN_classifier

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
# Feature Extraction Hooks
# ---------------------------------------------------------------------------------
feature_maps = {}

def hook_fn_conv1(module, input, output):
    """ Hook function to store feature maps for conv1 """
    feature_maps["conv1"] = output.detach()

def hook_fn_conv2(module, input, output):
    """ Hook function to store feature maps for conv2 """
    feature_maps["conv2"] = output.detach()

# Register hooks
model.conv1.register_forward_hook(hook_fn_conv1)
model.conv2.register_forward_hook(hook_fn_conv2)  # Added conv2

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

def visualize_feature_maps(feature_maps, layer_name, save_dir="feature_maps"):
    """ Saves and visualizes feature maps with layer name and filter index """
    os.makedirs(save_dir, exist_ok=True)

    activation = feature_maps[layer_name].cpu().numpy()  # Get feature maps
    num_filters = activation.shape[1]

    fig, axes = plt.subplots(2, min(6, num_filters), figsize=(15, 6))  # 2 rows for color & grayscale

    for i in range(min(6, num_filters)):
        # Color map (Viridis)
        axes[0, i].imshow(activation[0, i], cmap="viridis")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"{layer_name} Filter {i} (Color)")

        # Grayscale version
        axes[1, i].imshow(activation[0, i], cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title(f"{layer_name} Filter {i} (Grayscale)")

        # Save feature map with only layer name and filter index
        filename = f"{save_dir}/{layer_name}_filter_{i}.png"
        plt.imsave(filename, activation[0, i], cmap="gray")

    plt.show()
    print(f"Feature maps for {layer_name} saved in '{save_dir}/' directory.")


# Call visualization for both conv1 and conv2
if "conv1" in feature_maps:
    visualize_feature_maps(feature_maps, "conv1")

if "conv2" in feature_maps:
    visualize_feature_maps(feature_maps, "conv2")
