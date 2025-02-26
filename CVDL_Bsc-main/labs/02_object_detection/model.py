import torch
import torch.nn as nn
from torchvision.ops import roi_align


class FeatureExtractionNetwork(nn.Module):
    def __init__(
        self,
    ) -> None:
        super(FeatureExtractionNetwork, self).__init__()

    def forward(self, x):
        return x


class RegionProposalNetwork(nn.Module):
    def __init__(
        self,
    ) -> None:
        super(RegionProposalNetwork, self).__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        batch_size, _, h, w = x.shape

        boxes = []

        # returns random boundingboxes
        for _ in range(batch_size):
            x_min = torch.randint(0, w - 1, (1,))
            y_min = torch.randint(0, h - 1, (1,))
            x_max = torch.randint(x_min.item() + 1, w, (1,))
            y_max = torch.randint(y_min.item() + 1, h, (1,))

            box = torch.tensor(
                [x_min.item(), y_min.item(), x_max.item(), y_max.item()],
                dtype=torch.float,
            )
            boxes.append(box)

        return self.linear(torch.stack(boxes).to(x.device))


class ClassificationNetwork(nn.Module):
    def __init__(self, linear_size) -> None:
        super(ClassificationNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(linear_size, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x


class FasterRCNN(nn.Module):
    def __init__(
        self, feature_extraction_network, region_proposal_network, classifier, pool_size
    ) -> None:
        super(FasterRCNN, self).__init__()
        self.pool_size = pool_size

        self.fen = feature_extraction_network
        self.rpn = region_proposal_network
        self.clf = classifier

    def apply_roi_align(self, fmaps, bboxes):
        batch_indices = torch.arange(bboxes.size(0), device=bboxes.device).unsqueeze(1)
        rois = torch.cat([batch_indices.to(bboxes.dtype), bboxes], dim=1)
        aligned_features = roi_align(fmaps, rois, output_size=self.pool_size)
        return aligned_features

    def forward(self, x):
        # extract featuremaps
        fmaps = self.fen(x)
        # get bbox proposal
        bbox = self.rpn(fmaps)
        # pool fmaps
        pooled = self.apply_roi_align(fmaps, bbox)
        # classify
        return self.clf(pooled)
