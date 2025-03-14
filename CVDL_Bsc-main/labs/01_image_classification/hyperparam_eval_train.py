import torch
import torchvision.transforms.v2 as transforms
import optuna
from datasets import load_dataset
from model import NeuralNetwork, CNN_classifier
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

# ---------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------
IMG_SIZE = (430, 380)  # Image size for resizing during preprocessing
SEED = 42  # Random seed for reproducibility
DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)  # Automatically selects available device (GPU, Apple MPS, or CPU)
FNN_Flag = False  # Toggle between Fully Connected Neural Network (FNN) and CNN

# ---------------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(dataloader, model, loss_fn):
    """
    Evaluates the model on a given dataloader.
    
    Args:
        dataloader (DataLoader): Validation/Test dataloader.
        model (torch.nn.Module): Model to evaluate.
        loss_fn (torch.nn.Module): Loss function.
    
    Returns:
        tuple: (average loss, accuracy) on the dataset.
    """
    model.eval()
    test_loss, correct = 0, 0
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    for batch in dataloader:
        input, output = batch["img"].to(DEVICE), batch["class"].to(DEVICE)
        pred = model(input)
        test_loss += loss_fn(pred, output).item()
        correct += (pred.argmax(1) == output).type(torch.float).sum().item()
    
    return test_loss / num_batches, correct / num_samples


def train_one_epoch(dataloader, model, loss_fn, optimizer, epoch):
    """
    Trains the model for one epoch.
    
    Args:
        dataloader (DataLoader): Training dataloader.
        model (torch.nn.Module): Model to train.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epoch (int): Current epoch number.
    
    Returns:
        tuple: (average loss, accuracy) over the epoch.
    """
    model.train()
    train_loss, correct = 0, 0
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    for idx, batch in enumerate(dataloader):
        input, output = batch["img"].to(DEVICE), batch["class"].to(DEVICE)
        optimizer.zero_grad()
        pred = model(input)
        loss = loss_fn(pred, output)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        correct += (pred.argmax(1) == output).type(torch.float).sum().item()
        
        print(f"Epoch {epoch}, Step {idx+1}/{num_batches}: Loss={loss.item():.6f}")
    
    return train_loss / num_batches, correct / num_samples


def objective(trial):
    """
    Defines the objective function for Optuna hyperparameter tuning.
    
    Args:
        trial (optuna.trial.Trial): An Optuna trial object.
    
    Returns:
        float: Validation loss (to be minimized by Optuna).
    """
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RMSprop"])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    dropout_rate = trial.suggest_uniform("dropout", 0.2, 0.5)
    
    # Data Preparation
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
    data_loader_train = DataLoader(ds["train"], batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(ds["valid"], batch_size=1)
    
    # Model Initialization
    model = NeuralNetwork().to(DEVICE) if FNN_Flag else CNN_classifier(37).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
        "RMSprop": torch.optim.RMSprop,
    }[optimizer_name](model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training Loop
    EPOCHS = 5
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(data_loader_train, model, loss_fn, optimizer, epoch)
    
    # Evaluation
    val_loss, val_acc = evaluate(data_loader_valid, model, loss_fn)
    
    return val_loss  # Optuna minimizes the loss


def main():
    """
    Runs the Optuna study and trains the final model with the best parameters.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)  # Run 30 trials
    
    print("Best hyperparameters:", study.best_params)
    
    # Train final model with best hyperparameters
    best_params = study.best_params
    train_final_model(best_params)


def train_final_model(params):
    """
    Trains the final model using the best hyperparameters from Optuna.
    """
    print("Training final model with best hyperparameters...")
    
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
    data_loader_train = DataLoader(ds["train"], batch_size=params["batch_size"], shuffle=True)
    
    model = NeuralNetwork().to(DEVICE) if FNN_Flag else CNN_classifier(37).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
        "RMSprop": torch.optim.RMSprop,
    }[params["optimizer"]](model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    
    EPOCHS = 10  # More epochs for final training
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(data_loader_train, model, loss_fn, optimizer, epoch)
    
    torch.save(model.state_dict(), "hyper_model.pth")
    print("hyper_model saved.")

if __name__ == "__main__":
    torch.manual_seed(SEED)
    main()
