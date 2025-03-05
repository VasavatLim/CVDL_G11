from torch import nn

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


# UPDATED: CNN-based classifier implementation.
class CNN_classifier(nn.Module):
    def __init__(self, n_classes):
        super(CNN_classifier, self).__init__()
        # -- Convolutional Layers --
        # These layers extract spatial features from the images.
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        
        # -- Pooling Layer --
        # Adaptive average pooling converts feature maps to a fixed size regardless of input dimensions.
        self.adaptiveavgpooling = nn.AdaptiveAvgPool2d((11, 11)) 
        
        # -- Dense (Fully Connected) Layers --
        # Dense layers are necessary to integrate the spatial features learned by the conv layers
        # and to map these features to the final classification scores.
        self.neurallayer = nn.Linear(16 * 11 * 11, 256) # size = 11 x 11 (16 Feature maps)
        self.leakyrelu = nn.LeakyReLU()  # Leaky ReLU helps prevent vanishing gradients. 
        self.neurallayer2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.adaptiveavgpooling(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps.
        x = self.neurallayer(x)
        x = self.leakyrelu(x)
        logits = self.neurallayer2(x)
        return logits
