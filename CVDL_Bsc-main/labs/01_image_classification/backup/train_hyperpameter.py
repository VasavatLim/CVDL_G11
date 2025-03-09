import torch
import optuna
import torchvision.transforms.v2 as transforms
from datasets import load_dataset
from model import NeuralNetwork, CNN_classifier
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

# ---------------------------------------------------------------------------------
# Global Constants
# ---------------------------------------------------------------------------------
IMG_SIZE = (128, 128)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ---------------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(dataloader, model, loss_fn):
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    for batch in dataloader:
        input, output = batch["img"].to(DEVICE), batch["class"].to(DEVICE)
        pred = model(input)
        test_loss += loss_fn(pred, output).item()
        correct += (pred.argmax(1) == output).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= num_samples
    return test_loss, correct  # Return loss and accuracy

def train_one_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch in dataloader:
        input, output = batch["img"].to(DEVICE), batch["class"].to(DEVICE)

        # Forward
        pred = model(input)
        loss = loss_fn(pred, output)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ---------------------------------------------------------------------------------
# Objective Function for Optuna
# ---------------------------------------------------------------------------------
def objective(trial):
    # Sample hyperparameters
    model_type = trial.suggest_categorical("model", ["FNN", "CNN"])
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)  # Use suggest_float with log=True
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])  # Discrete choice
    optimizer_type = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.5)  # Dropout for regularization
    num_filters = trial.suggest_categorical("num_filters", [16, 32, 64])  # CNN filter size
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)  # L2 regularization
    scheduler_type = trial.suggest_categorical("scheduler", ["StepLR", "ReduceLROnPlateau"])

    # Load dataset
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

    # Create DataLoaders
    train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(ds["valid"], batch_size=1)

    # Initialize model
    if model_type == "FNN":
        model = NeuralNetwork().to(DEVICE)
    else:
        model = CNN_classifier(37).to(DEVICE)  # Tune CNN with 37 output classes

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer selection
    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    if scheduler_type == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    else:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    # Train for a few epochs
    EPOCHS = 5  # Short runs for tuning
    for epoch in range(EPOCHS):
        train_one_epoch(train_loader, model, loss_fn, optimizer)
        val_loss, _ = evaluate(valid_loader, model, loss_fn)
        scheduler.step(val_loss if scheduler_type == "ReduceLROnPlateau" else None)

    # Return validation loss for Optuna to minimize
    return val_loss

# ---------------------------------------------------------------------------------
# Run Optuna Hyperparameter Optimization
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(SEED)

    # Run optimization (30 trials)
    study = optuna.create_study(direction="minimize")  # Minimize validation loss
    study.optimize(objective, n_trials=30)  # Run 30 trials

    # Print best hyperparameters
    print("\nBest hyperparameters:", study.best_params)

    # Train final model with best hyperparameters
    best_params = study.best_params
    best_model_type = best_params["model"]
    best_lr = best_params["learning_rate"]
    best_batch_size = best_params["batch_size"]
    best_optimizer_type = best_params["optimizer"]
    best_dropout = best_params["dropout"]
    best_num_filters = best_params["num_filters"]
    best_weight_decay = best_params["weight_decay"]
    best_scheduler_type = best_params["scheduler"]

    # Reload dataset
    train_loader = DataLoader(ds["train"], batch_size=best_batch_size, shuffle=True)
    valid_loader = DataLoader(ds["valid"], batch_size=1)

    # Initialize best model
    if best_model_type == "FNN":
        best_model = NeuralNetwork().to(DEVICE)
    else:
        best_model = CNN_classifier(37).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    if best_optimizer_type == "SGD":
        best_optimizer = torch.optim.SGD(best_model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
    elif best_optimizer_type == "Adam":
        best_optimizer = torch.optim.Adam(best_model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
    else:
        best_optimizer = torch.optim.AdamW(best_model.parameters(), lr=best_lr, weight_decay=best_weight_decay)

    writer = SummaryWriter(log_dir=f"runs/Best_Model")

    FINAL_EPOCHS = 10
    for epoch in range(FINAL_EPOCHS):
        train_one_epoch(train_loader, best_model, loss_fn, best_optimizer)
        val_loss, val_acc = evaluate(valid_loader, best_model, loss_fn)
        writer.add_scalar("Loss/Final_Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Final_Validation", val_acc, epoch)

    # Save best model
    torch.save(best_model.state_dict(), "best_model.pth")
    print("\n✅ Best model saved as best_model.pth")

    writer.close()
