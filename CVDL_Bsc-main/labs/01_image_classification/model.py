from torch import nn


# currently its a fully connected feed forward layer with one hidden layer with 2048 neurons
# 128*128*3 (x pixel times y pixel times rgb channels ) inpout neurons and 37 output (corresponding to number of cclasses)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.ffnn = nn.Sequential(
            nn.Linear(128 * 128 * 3, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 37),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.ffnn(x)
        return logits


class CNN_classifier(nn.Module):
    def __init__(self, n_classes):
        super(CNN_classifier, self).__init__()
        # in_channels = 3 because 3 rgb channels, 16 out = umbers of feature maps extracted, stride =1, padding = 1 adds 1 pixel padding, for zero padding
        # we could later introduce 1 paddinfg in the first layer to FORCE features activation in the first  layer and prevent
        # neurons from dying paddin (optional)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1
        )
        # output size = (input+2*padding-kernel size/stride) +1
        # check if 1 pixel is sufficent
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2
        )  # set padding to 2 to maintain 22x22 size
        self.adaptiveavgpooling = nn.AdaptiveAvgPool2d((11, 11))
        # self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # assuming 16 feature maps and 22*22 inital image size (reduced half trough pooling)
        self.neurallayer = nn.Linear(16 * 11 * 11, 256)
        self.leakyrelu = nn.LeakyReLU()  # leaky relu to prevent vanishing gradient
        self.neurallayer2 = nn.Linear(256, n_classes)

    def forward(self, x):
        network = self.conv1(x)
        network = self.leakyrelu(
            network
        )  # applying a relu here else its a linear transformation
        network = self.conv2(network)
        network = self.adaptiveavgpooling(network)
        network = network.view(network.size(0), -1)
        network = self.neurallayer(network)
        network = self.leakyrelu(network)
        return self.neurallayer2(network)