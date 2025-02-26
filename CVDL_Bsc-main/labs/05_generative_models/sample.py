import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2.functional as F
from matplotlib.widgets import Slider
from model import SDUNet
from noise_scheduler import LinearNoisescheduler
from torchvision.utils import make_grid
from tqdm import tqdm

# ---------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------
# sampling params
IMG_SIZE = (64, 64)
SEED = 42
MODEL_IN = "model.pth"
PLOT_HIST = True
NUM_COLS = 4
TEMP = 0.0

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


def denormalize(batch):
    mean = torch.tensor([-0.0325, -0.1279, -0.2258])[:, None, None]
    std = torch.tensor([0.5159, 0.5002, 0.5063])[:, None, None]

    return batch * std.to(batch.device) + mean.to(batch.device)


def normalize(batch):
    mean = [-0.0325, -0.1279, -0.2258]
    std = [0.5159, 0.5002, 0.5063]

    return F.normalize(batch, mean=mean, std=std)


def range_n11(x):
    return (x * 2) - 1


def range_01(x):
    return (x + 1.0) / 2.0


def sample_noise(img_size, model, sched, ncols, seed, history, temp=0.0):
    # helper vars
    device = sched.device
    # random noise
    gen = torch.manual_seed(seed)
    xt = torch.randn((ncols**2, model.in_channels, *img_size), generator=gen).to(device)
    # tmp result storage
    h = sched.num_timesteps if history else 1
    res = torch.zeros((h, *make_grid(xt, nrow=ncols).size()))

    # sample img
    with torch.no_grad():
        for t in tqdm(range(sched.num_timesteps), desc="sampling", leave=False):
            # reverse t
            t = sched.num_timesteps - t - 1
            # predict noise
            noise_pred = model(xt, torch.as_tensor(t).unsqueeze(0).to(device))
            # Use scheduler to get x0 and xt-1
            xt, x0 = sched.sample_prev(xt, noise_pred, torch.as_tensor(t).to(device))
            # add back noise
            if temp > 0:
                xt = xt + temp * (sched.a_cp[t] ** 2) * torch.randn_like(xt)
            # scale and save x0
            imgs = denormalize(xt)
            imgs = range_01(imgs)
            imgs = torch.clamp(imgs, 0.0, 1.0)
            # create grid
            grid = make_grid(imgs, nrow=ncols).to(device)
            t = t if history else 0
            res[t] = grid

    return torch.flip(res, [0]).detach().cpu()


def plot_img(img):
    img = img.cpu()
    # Plotting the first image
    fig, ax = plt.subplots()
    current_image = ax.imshow(img[0].permute(1, 2, 0).numpy())

    # slider if multiple
    if img.size(0) > 1:
        # Create the slider
        plt.subplots_adjust(bottom=0.25)
        ax_slider = plt.axes([0.1, 0.05, 0.8, 0.05])
        slider = Slider(
            ax_slider, "Timestep", 0, sched.num_timesteps - 1, valinit=0, valfmt="%0.0f"
        )

        # Update function for the slider
        def update(val):
            index = int(slider.val)
            current_image.set_data(img[index].permute(1, 2, 0).numpy())
            fig.canvas.draw_idle()

        slider.on_changed(update)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------
    # model & scheduler
    # ----------------------------------------------------------------------------------
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
    weights = torch.load(MODEL_IN, map_location=DEVICE)
    model.load_state_dict(weights)

    sched = LinearNoisescheduler(
        beta_start=BETA_START, beta_end=BETA_END, num_timesteps=NUM_T, device=DEVICE
    )

    # ----------------------------------------------------------------------------------
    # sampling
    # ----------------------------------------------------------------------------------
    # generate img
    img = sample_noise(
        img_size=IMG_SIZE,
        model=model,
        sched=sched,
        ncols=NUM_COLS,
        seed=SEED,
        history=PLOT_HIST,
        temp=TEMP,
    )
    # plot img
    plot_img(img)
