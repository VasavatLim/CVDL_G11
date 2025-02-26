import torch
import torchvision.transforms.v2 as transforms
from datasets import load_dataset
from model import SDUNet
from noise_scheduler import LinearNoisescheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

# ---------------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------------
# training process
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 30
IMG_SIZE = (64, 64)
SEED = 42

# model
IN_CHANNELS = 3
OUT_CHANNELS = 3
CHANNELS_BASE = 32
MULTIPLIERS = (1, 2)
ATTN_LEVELS = (0, 1)
N_LVLBLOCKS = 2
T_EMB_DIM = 64
LVLBLOCKTYPE = "cov"
ATTBLOCKTYPE = "self"

# scheduler
BETA_START = 0.0001
BETA_END = 0.02
NUM_T = 1000

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


def train_one_epoch(
    writer, step, dataloader, model, noise_scheduler, loss_fn, optimizer
):
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    model.train()
    for idx, batch in enumerate(dataloader):
        x0 = batch["img"].to(DEVICE)
        # sample random noise
        e = torch.randn_like(x0, device=DEVICE)

        # sample random timesteps
        t = noise_scheduler.sample_timesteps(x0.size(0))

        # apply noise
        xt = noise_scheduler.add_noise(x0, e, t)

        # forward
        e_pred = model(xt, t)
        loss = loss_fn(e_pred, e)
        train_loss += loss.item()

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (idx + 1) * len(x0)
        print(f"loss: {loss:>7f}  [{current:>5d}/{num_samples:>5d}]")

    # compute metrics
    train_loss /= num_batches

    # logging
    writer.add_scalar("Loss/Train", train_loss, step)
    print(f"Train Error: \n Avg loss: {train_loss:>8f} \n")


def range_n11(x):
    return (x * 2) - 1


def main():
    # -----------------------------------------------------------------------------
    # data
    # -----------------------------------------------------------------------------
    transform_img = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Resize(IMG_SIZE, antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Lambda(range_n11),
            transforms.Normalize(
                mean=[-0.0325, -0.1279, -0.2258], std=[0.5159, 0.5002, 0.5063]
            ),
        ]
    )

    def transform(samples):
        samples["img"] = [transform_img(img) for img in samples["img"]]
        return samples

    # load dataset & apply transform
    ds = load_dataset("cvdl/catfaces")
    ds = ds.select_columns(["img"])
    ds = ds.with_transform(transform)

    # Create data loaders.
    data_loader = DataLoader(
        ds["train"],
        batch_size=BATCH_SIZE,
    )

    # ---------------------------------------------------------------------------------
    # scheduler
    # ---------------------------------------------------------------------------------
    scheduler = LinearNoisescheduler(
        beta_start=BETA_START, beta_end=BETA_END, num_timesteps=NUM_T, device=DEVICE
    )

    # ---------------------------------------------------------------------------------
    # model & optimizer
    # ---------------------------------------------------------------------------------
    model = SDUNet(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        channels_base=CHANNELS_BASE,
        multipliers=MULTIPLIERS,
        attn_levels=ATTN_LEVELS,
        n_lvlblocks=N_LVLBLOCKS,
        t_emb_dim=T_EMB_DIM,
        lvlblock_type=LVLBLOCKTYPE,
        attblock_type=ATTBLOCKTYPE,
    ).to(DEVICE)
    model.train()

    # Specify training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    # ---------------------------------------------------------------------------------
    # logging
    # ---------------------------------------------------------------------------------
    writer = SummaryWriter()

    # ---------------------------------------------------------------------------------
    # training
    # ---------------------------------------------------------------------------------
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_one_epoch(
            writer, t + 1, data_loader, model, scheduler, criterion, optimizer
        )
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
