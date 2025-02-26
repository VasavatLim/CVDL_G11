import matplotlib.pyplot as plt  # noqa: F401
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from datasets import load_dataset
from torchvision import tv_tensors
from torchvision.utils import draw_bounding_boxes


def plot(img, bbox, pred):
    img_gt = draw_bounding_boxes(img.clone(), bbox, colors="red", width=2)  # noqa: F481
    img_pred = draw_bounding_boxes(img.clone(), pred, colors="green", width=2)  # noqa: F481

    # PLOT IMAGES HERE


def main(modelpath):
    # -----------------------------------------------------------------------------
    # models
    # -----------------------------------------------------------------------------
    fen = None  # YOUR FEATURE EXTRACTION NETWORK HERE

    rpn = None  # YOUR REGION PROPOSAL NETWORK HERE
    rpn.load_state_dict(torch.load(modelpath, map_location="cpu"))
    model = nn.Sequential(fen, rpn)
    model.eval()

    # -----------------------------------------------------------------------------
    # data
    # -----------------------------------------------------------------------------
    transform_img = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Resize((380, 430), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )
    transform_box = transforms.Resize((380, 430))

    def transform(samples):
        for idx in range(len(samples["img"])):
            img, bbox = samples["img"][idx], samples["bbox"][idx]
            bbox = tv_tensors.BoundingBoxes(
                data=bbox,
                format="XYXY",
                canvas_size=(img.size[1], img.size[0]),
            )
            samples["img"][idx] = transform_img(img)
            samples["bbox"][idx] = transform_box(bbox)
        return samples

    # load dataset & apply transform
    ds = load_dataset("cvdl/oxford-pets")
    ds = ds.select_columns(["img", "bbox"])
    ds = ds.with_transform(transform)

    # -----------------------------------------------------------------------------
    # visualize
    # -----------------------------------------------------------------------------
    for sample in ds["test"]:
        img, bbox = sample["img"], sample["bbox"]
        pred = model(img.unsqueeze(0))
        plot(img, bbox, pred)


if __name__ == "__main__":
    modelpath = "PATH-TO-SAVED-RPN"
    main(modelpath)
