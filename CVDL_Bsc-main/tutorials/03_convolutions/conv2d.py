import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.io import read_image


def convolve(path_img, kernel, padding, stride):
    # read img
    img = read_image(path_img).unsqueeze(0) / 255

    # construct kernel
    kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(img.size(1), 1, 1, 1)

    # apply conv2d
    res = F.conv2d(img, kernel, groups=img.size(1), padding=padding, stride=stride)
    print(res.size())

    # display image
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(img.squeeze(0).permute(1, 2, 0).numpy())
    ax1.axis("off")
    ax1.set_title("original")
    ax2.imshow(res.squeeze(0).permute(1, 2, 0).numpy())
    ax2.axis("off")
    ax2.set_title("convolved")

    plt.show()


if __name__ == "__main__":
    path_img = os.path.join(os.path.dirname(__file__), "img/cat.png")
    kernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]

    convolve(path_img=path_img, kernel=kernel, padding=0, stride=1)
