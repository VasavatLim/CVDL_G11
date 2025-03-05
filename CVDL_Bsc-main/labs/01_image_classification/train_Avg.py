import torch
import torchvision.transforms.v2 as transforms
from datasets import load_dataset
from model import NeuralNetwork  # Import the Original model. (FFNN)
from model import CNN_classifier  # Import the new CNN model. (CNN)
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

# ---------------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------------
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 5
# IMG_SIZE will be computed dynamically from the dataset's average resolution.
SEED = 42

MODEL_OUT = "model.pth"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ---------------------------------------------------------------------------------
# HELPER FUNCTIONS (Evaluation & Training Loop)
# ---------------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(writer, step, dataloader, model, loss_fn):
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    for batch in dataloader:
        inputs, targets = batch["img"].to(DEVICE), batch["class"].to(DEVICE)
        pred = model(inputs)
        test_loss += loss_fn(pred, targets).item()
        correct += (pred.argmax(1) == targets).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= num_samples

    writer.add_scalar("Loss/Test", test_loss, step)
    writer.add_scalar("Accuracy/Test", correct, step)
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_one_epoch(writer, step, dataloader, model, loss_fn, optimizer):
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    model.train()
    for idx, batch in enumerate(dataloader):
        inputs, targets = batch["img"].to(DEVICE), batch["class"].to(DEVICE)
        pred = model(inputs)
        loss = loss_fn(pred, targets)
        train_loss += loss.item()
        correct += (pred.argmax(1) == targets).type(torch.float).sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_val = loss.item()
        current = (idx + 1) * len(inputs)
        print(f"loss: {loss_val:>7f}  [{current:>5d}/{num_samples:>5d}]")

    train_loss /= num_batches
    correct /= num_samples

    writer.add_scalar("Loss/Train", train_loss, step)
    writer.add_scalar("Accuracy/Train", correct, step)
    print(f"Train Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")


def main():
    # ===========================
    # PREPARE THE DATA
    # ===========================
    # Load the dataset without transformation initially to compute average resolution.
    ds = load_dataset("cvdl/oxford-pets")
    ds = ds.select_columns(["img", "class"])

    # --- Compute Average Resolution ---
    # Calculate average width and height from the training set to determine the target image size.
    print("Computing average resolution from the training set...")
    total_width, total_height, count = 0, 0, 0
    for sample in ds["train"]:
        w, h = sample["img"].size  # PIL Image: (width, height)
        total_width += w
        total_height += h
        count += 1
    avg_width = int(total_width / count)
    avg_height = int(total_height / count)
    print(f"Average resolution: {avg_width} x {avg_height}")

    # Use the computed average resolution as the target image size.
    IMG_SIZE = (avg_width, avg_height)

    # --- Define Data Transformations ---
    # Resize images to IMG_SIZE, convert them to tensors, and scale pixel values.
    transform_img = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    def transform(samples):
        samples["img"] = [transform_img(img) for img in samples["img"]]
        samples["class"] = [torch.tensor(c) for c in samples["class"]]
        return samples

    # Apply the transformation to the dataset.
    ds = ds.with_transform(transform)

    # Create data loaders.
    data_loader_train = DataLoader(
        ds["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    data_loader_valid = DataLoader(
        ds["valid"],
        batch_size=1,
    )

    # ===========================
    # BUILD THE MODEL (for training)
    # ===========================
    # Use the CNN_classifier for training. The CNN extracts spatial features,
    # and the dense layers map these features to the final class scores.
    model = CNN_classifier(n_classes=37).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # ---------------------------------------------------------------------------------
    # Logging and Training Loop
    # ---------------------------------------------------------------------------------
    writer = SummaryWriter()

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_one_epoch(writer, t + 1, data_loader_train, model, loss_fn, optimizer)
        evaluate(writer, t + 1, data_loader_valid, model, loss_fn)
    writer.close()
    print("Done!")

    # Save the trained model.
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Saved PyTorch Model State to {MODEL_OUT}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    main()
