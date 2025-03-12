
import torch
import torch.nn as nn
import torchvision.models as models  # ✅ FIX: Import torchvision.models
# ===========================
# BUILD THE MODEL
# ===========================

# This model uses a Convolutional Neural Network (CNN) architecture for image classification.
# The model includes:
# 1. Convolutional Layers: Extract spatial features from the input images.
# 2. Pooling Layer: Reduces the spatial dimensions and makes the model invariant to small translations.
# 3. Dense (Fully Connected) Layers: Integrate the extracted features into a final prediction vector.
#    Dense layers are necessary because they learn non-linear combinations of the features
#    from the convolutional layers, mapping these features to the output class scores.

# Fully connected network (original approach) is kept here for reference.

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
    def __init__(self, n_classes, dropout=0.2):
        super(CNN_classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max Pooling

        self.adaptiveavgpooling = nn.AdaptiveAvgPool2d((10, 10))
        self.neurallayer = nn.Linear(16 * 10 * 10, 512)
        self.neurallayer2 = nn.Linear(512, 256)
        self.neurallayer3 = nn.Linear(256, n_classes)
        self.leakyrelu = nn.LeakyReLU()
        self.bnorm = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        network = self.conv1(x)
        network = self.bnorm(network)
        network = self.leakyrelu(network)
        network = self.pool(network)
        network = self.conv2(network)
        network = self.bnorm(network)
        network = self.leakyrelu(network)
        network = self.adaptiveavgpooling(network)
        network = network.view(network.size(0), -1)
        network = self.neurallayer(network)
        network = self.leakyrelu(network)
        network = self.dropout(network)

        network = self.neurallayer2(network)
        network = self.leakyrelu(network)
        network = self.dropout(network)
        return self.neurallayer3(network)
#Trainable parameters in FFNN: 104937509
#Trainable parameters in CNN: 967445

