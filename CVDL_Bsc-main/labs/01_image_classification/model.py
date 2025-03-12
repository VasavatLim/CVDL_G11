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
    def __init__(self, n_classes,dropout = 0.2):
        super(CNN_classifier, self).__init__()
        # in_channels = 3 because 3 rgb channels, 16 out = umbers of feature maps extracted, stride =1, padding = 1 adds 1 pixel padding, for zero padding
        # we could later introduce 1 paddinfg in the first layer to FORCE features activation in the first  layer and prevent
        # neurons from dying paddin (optional)
        # feature extractio n
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1
        )
        # output size = (input+2*padding-kernel size/stride) +1
        # check if 1 pixel is sufficent
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2
        )  # set padding to 2 to maintain 22x22 size
        # we enforce the same output of 11 x 11 here by using adaprtive avg pooling
        self.adaptiveavgpooling = nn.AdaptiveAvgPool2d((25, 25))
        # self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # assuming 16 feature maps and 22*22 inital image size (reduced half trough pooling)
        # classification
        self.neurallayer = nn.Linear(16 * 25 * 25, 1024)
        self.neurallayer2 = nn.Linear(1024, 1024)

        self.leakyrelu = nn.LeakyReLU()  # leaky relu to prevent vanishing gradient
        self.neurallayer3 = nn.Linear(1024, n_classes)
        self.bnorm = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(dropout)  # Prevenfdt overdfitting


    def forward(self, x):
        network = self.conv1(x)
        network = self.leakyrelu(
            network
        )  # applying a relu here else its a linear transformation
        network = self.conv2(network)
        network = self.bnorm(network)

        network = self.leakyrelu(network)
        network = self.adaptiveavgpooling(network)
        network = network.view(network.size(0), -1)
        network = self.neurallayer(network)
        network = self.leakyrelu(network)
        network = self.neurallayer2(network)
        network = self.leakyrelu(network)
        network = self.neurallayer2(network)
        network = self.leakyrelu(network) #//maybe later add a dropout layer
        network = self.dropout(network)
        return self.neurallayer3(network)
#Trainable parameters in FFNN: 104937509
#Trainable parameters in CNN: 11335445
