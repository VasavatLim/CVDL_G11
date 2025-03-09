import torch
from model import CNN_classifier
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (430, 380)
MODEL_PATHS = ["hyper_model.pth", "model_cnn.pth"]  # Comparing tuned and default models

# ---------------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------------
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
data_loader_valid = DataLoader(ds["valid"], batch_size=1)

# ---------------------------------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(model, dataloader):
    """
    Evaluates the model and computes accuracy, precision, recall, and F1-score.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        dataloader (DataLoader): Validation data.

    Returns:
        dict: Contains accuracy, precision, recall, and F1-score.
    """
    model.eval()
    correct = 0
    num_samples = len(dataloader.dataset)

    all_preds = []
    all_labels = []

    for batch in dataloader:
        input, output = batch["img"].to(DEVICE), batch["class"].to(DEVICE)
        pred = model(input)
        predicted_labels = pred.argmax(1)

        correct += (predicted_labels == output).type(torch.float).sum().item()

        all_preds.extend(predicted_labels.cpu().numpy())
        all_labels.extend(output.cpu().numpy())

    accuracy = (correct / num_samples) * 100
    precision = precision_score(all_labels, all_preds, average="macro") * 100
    recall = recall_score(all_labels, all_preds, average="macro") * 100
    f1 = f1_score(all_labels, all_preds, average="macro") * 100

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }

# ---------------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------------
for model_path in MODEL_PATHS:
    print(f"\nEvaluating {model_path}...")
    
    model = CNN_classifier(37).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    
    metrics = evaluate(model, data_loader_valid)
    
    print(f"Model: {model_path}")
    print(f"Accuracy: {metrics['Accuracy']:.2f}%")
    print(f"Precision: {metrics['Precision']:.2f}%")
    print(f"Recall: {metrics['Recall']:.2f}%")
    print(f"F1-Score: {metrics['F1-Score']:.2f}%\n")
