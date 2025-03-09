import torch
import torchvision.transforms.v2 as transforms
from datasets import load_dataset
from model import CNN_classifier
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import random

# ---------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------
IMG_SIZE = (430, 380)
SEED = 42
DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
#MODEL_PATH = "best_model.pth"
MODEL_PATH = "model_cnn.pth"

# Set random seeds for reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------------------
def load_model():
    model = CNN_classifier(37)  # Ensure architecture matches training
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ---------------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------------
def get_test_dataloader():
    transform_img = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(IMG_SIZE, antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    def transform(samples):
        samples["img"] = [transform_img(img) for img in samples["img"]]
        samples["class"] = [torch.tensor(c) for c in samples["class"]]
        return samples

    ds = load_dataset("cvdl/oxford-pets").select_columns(["img", "class"]).with_transform(transform)
    data_loader_test = DataLoader(ds["valid"], batch_size=32, shuffle=False)  # Match train.py split
    return data_loader_test

# ---------------------------------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------------------------------
def evaluate(model, dataloader):
    loss_fn = nn.CrossEntropyLoss()
    test_loss, correct = 0, 0
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    y_true, y_pred = [], []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input, output = batch["img"].to(DEVICE), batch["class"].to(DEVICE)
            pred = model(input)
            loss = loss_fn(pred, output).item()
            correct_batch = (pred.argmax(1) == output).type(torch.float).sum().item()

            test_loss += loss
            correct += correct_batch

            y_true.extend(output.cpu().numpy())
            y_pred.extend(pred.argmax(1).cpu().numpy())

            print(f"Batch {i+1}/{num_batches}: Loss={loss:.4f}, Correct={correct_batch}/{len(output)}")

    test_loss /= num_batches
    accuracy = correct / num_samples
    print(f"Test Accuracy: {accuracy:.4f}, Avg Loss: {test_loss:.6f}")
    
    return y_true, y_pred, accuracy, test_loss

# ---------------------------------------------------------------------------------
# Generate Classification Report & Confusion Matrix
# ---------------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    print("Loading Model...")
    model = load_model()
    
    print("Loading Test Data...")
    test_dataloader = get_test_dataloader()
    
    print("Evaluating Model...")
    y_true, y_pred, accuracy, test_loss = evaluate(model, test_dataloader)
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    print("Plotting Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred)
