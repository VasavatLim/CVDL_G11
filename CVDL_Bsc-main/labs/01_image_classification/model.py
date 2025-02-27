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
