import torch
import torchvision.transforms.v2 as transforms
from datasets import load_dataset
from model import FeatureExtractionNetwork, RegionProposalNetwork
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.tv_tensors import BoundingBoxes

# ---------------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------------
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 5
IMG_SIZE = (380, 430)
SEED = 42

# helper vars
MODEL_OUT = "model.pth"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ---------------------------------------------------------------------------------
# helper funcs
# ---------------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(writer, step, dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0
    model.eval()
    for batch in dataloader:
        input, output = batch["img"].to(DEVICE), batch["bbox"].to(DEVICE)

        pred = model(input)
        test_loss += loss_fn(pred, output).item()

    # compute metrics
    test_loss /= num_batches

    # logging
    writer.add_scalar("Loss/Test", test_loss, step)
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def train_one_epoch(writer, step, dataloader, model, loss_fn, optimizer):
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    model.train()
    for idx, batch in enumerate(dataloader):
        input, output = batch["img"].to(DEVICE), batch["bbox"].to(DEVICE)

        # forward
        pred = model(input)
        loss = loss_fn(pred, output)
        train_loss += loss.item()

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (idx + 1) * len(input)
        print(f"loss: {loss:>7f}  [{current:>5d}/{num_samples:>5d}]")

    # compute metrics
    train_loss /= num_batches

    # logging
    writer.add_scalar("Loss/Train", train_loss, step)
    print(f"Train Error: \n Avg loss: {train_loss:>8f} \n")


def main():
    # -----------------------------------------------------------------------------
    # data
    # -----------------------------------------------------------------------------
    transform_img = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )
    transform_box = transforms.Resize(IMG_SIZE)

    def transform(samples):
        for idx in range(len(samples["img"])):
            img, bbox = samples["img"][idx], samples["bbox"][idx]
            bbox = BoundingBoxes(
                data=bbox,
                format="XYXY",
                canvas_size=(img.size[1], img.size[0]),
                dtype=torch.float32,
            )
            samples["img"][idx] = transform_img(img)
            samples["bbox"][idx] = transform_box(bbox).squeeze(0)
        return samples

    # load dataset & apply transform
    ds = load_dataset("cvdl/oxford-pets")
    ds = ds.select_columns(["img", "bbox"])
    ds = ds.with_transform(transform)

    # Create data loaders.
    data_loader_train = DataLoader(
        ds["train"],
        batch_size=BATCH_SIZE,
    )
    data_loader_valid = DataLoader(
        ds["valid"],
        batch_size=1,
    )

    # ---------------------------------------------------------------------------------
    # model & optimizer
    # ---------------------------------------------------------------------------------
    model = nn.Sequential(FeatureExtractionNetwork(), RegionProposalNetwork()).to(
        DEVICE
    )
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # ---------------------------------------------------------------------------------
    # logging
    # ---------------------------------------------------------------------------------
    writer = SummaryWriter()

    # ---------------------------------------------------------------------------------
    # training
    # ---------------------------------------------------------------------------------
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_one_epoch(writer, t + 1, data_loader_train, model, loss_fn, optimizer)
        evaluate(writer, t + 1, data_loader_valid, model, loss_fn)
    writer.close()
    print("Done!")

    # ---------------------------------------------------------------------------------
    # save model
    # ---------------------------------------------------------------------------------
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Saved PyTorch Model State to {MODEL_OUT}")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    main()
