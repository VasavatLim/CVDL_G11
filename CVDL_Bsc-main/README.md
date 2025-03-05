# 01 Image Classification Lab

## Introduction
In this lab, you will build and experiment with an image classification model using the [Oxford Pets dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). This dataset consists of images of varying resolutions, each depicting either a dog or a cat. You will explore neural network architectures, train models, analyze their performance, and gain insights into how Convolutional Neural Networks (CNNs) process image features.


## Task I: Understand the Provided Python Scripts (1P)
Your first task is to analyze the provided Python scripts to understand how the initial model works. Focus on the following key aspects:

1. **Network Architecture**  
   - Identify and describe the layers of the neural network.

   Ans. This neural network in model.py is a Feedforward Neural Network (FFNN) designed for classification. This network processes input images of size 128 x 128 with 3 color channels(RGB). The first layer is flatten layer which converts 3D input image of shape (3,128,128) into 1D tensor of shape (128 x 128 x 3 = 49,152). The next layer is first fully connected layer that has input as 49,152 flatten image pixels then has output as 2,048 neurons. Then there is ReLU activation and second fully connected layer that maintains 2,048 neurons.Then there is another ReLU activation and ends with output layer reducing the features to 37 neurons for 37-class classification problem. 
   
2. **Optimizer**  
   - Explain the choice of the optimizer and its role in training the model.

   Ans. The choice of optimizer in train.py is Stochastic Gradient Descent (SGD) which is a widely used algorithm for training neural networks. SGD works well with the large-scale datasets. It allows for fine control over convergence with a learning rate.

3. **Loss Function**  
   - Review the loss function used and explain why it is appropriate for this classification task.

   Ans. In train.py, we use CrossEntropyLoss for loss function since it is suitable for multi-class classification where outputs represent class probabilities. 

4. **TensorBoard Logging**  
   - Examine how TensorBoard is integrated into the training process.  

   Ans. TensorBoard is integrated using SummaryWriter from torch.utils.tensorboard. First, the SummaryWriter is initialized at the beginning of training for TensorBoard to create log directory to store training metrics. Then during each epoch, we log the training loss and training accuracy. After each epoch, the model is evaluated on the validation dataset about the loss and accuraty and the results are logged. This process can help check tracking overfitting. Lastly, at the end of training, the writer is closed and all logs are saved.

   - Describe the benefits of using TensorBoard for monitoring training progress.

   Ans. Benefits of using TensorBoard are visualizing loss and accuracy trends over epochs, help diagnosing overfitting or underfitting, and enabling easy comparison of different training runs.


## Task II: Familiarize Yourself with the Dataset (1P)
The Oxford Pets dataset will serve as the foundation for model training. In this task, you will explore it in detail:

1. **Dataset Overview**  
   - Analyze the variety of images in the dataset (e.g., resolution, composition, etc.).
   
   Ans.Total images: 7390

   For sample of 100 images,
Resolution Analysis:
Mean resolution: [423.35 385.99]
Min resolution: [140 134]
Max resolution: [600 531]

Aspect Ratio Analysis:
Mean aspect ratio: 1.1573654417783406
Min aspect ratio: 0.638
Max aspect ratio: 1.6181229773462784

Color Composition Analysis:
Mean RGB color composition: [103.09501853 115.27062977 121.3184631 ]
   
   - Describe the types of information you can extract from the dataset (e.g., labels, image dimensions, etc.).

   Ans. image dimensions: width and height, aspect ratio
   color composition
   category labels (breed of each pet)
   predefined splits (trainval.txt and test.txt)
   class distribution


2. **Insights Gained**  
   - Provide a summary of your observations about the dataset's structure and content.

   Ans. The dataset consists of images and corresponding annotations. It is a 37 category pet dataset with roughly 200 images for each class.
   
   The images have a large variations in scale, pose and lighting. All images have an associated ground truth annotation of breed, head ROI, and pixel level trimap segmentation. 
   
## Task III: Implement a CNN (2P)
Your goal in this task is to design and implement a Convolutional Neural Network (CNN) with dense (fully connected) layers for image classification. This involves the following steps:

1. **Build the Model**  
    - Specify the layers of your CNN, such as convolutional layers, pooling layers, and dense layers.
    
    Ans.
Convolutional Layer 1
Filters: 32, Kernel: 3×3, Activation: ReLU
Followed by MaxPooling (2×2)

Convolutional Layer 2
Filters: 64, Kernel: 3×3, Activation: ReLU
Followed by MaxPooling (2×2)

Convolutional Layer 3
Filters: 128, Kernel: 3×3, Activation: ReLU
Followed by MaxPooling (2×2)

Flattening Layer
Converts feature maps into a 1D vector

Fully Connected (Dense) Layers
Dense Layer 1: 256 neurons, Activation: ReLU
Dropout Layer: Prevents overfitting
Output Layer: Softmax Activation (for multi-class classification)
      
    - Clearly explain why dense layers are necessary for the classification process.

    Ans.The final dense layers are necessary to often use with softmax activation to produce probabilities for each class. 


2. **Prepare the Data**
    - Resize the images to ensure uniform input dimensions.  
    - **Hint:** Calculate the average height and width of the images in the dataset to choose appropriate resize parameters.  

   Ans. Average Image Size: 436x391


## Task IV: Compare Models (2P)
Compare the performance of your implemented CNN to the original provided network. Use the following steps to guide your analysis:

1. **Train Both Models**  
   - Train the provided network and your custom CNN on the dataset.

2. **Analyze the Results**  
   - Count the total number of trainable parameters in each model.  
   - Discuss how the number of parameters impacts the performance (e.g., accuracy, training time).


## Task V: Visualize Feature Maps (2P)
In this task, you will investigate the inner workings of your CNN by visualizing feature maps generated at different layers of the network.

1. **Modify Your CNN**  
   - Add functionality to save and retrieve outputs (feature maps) from selected convolutional layers.

2. **Visualize Feature Maps**  
   - Visualize feature maps from the first and second convolutional layers.  
   - **Hint:** Represent the feature maps as grayscale images to interpret the patterns detected at each depth.


## Task VI: (Optional) Further Exploration (2P)
For further investigation and to deepen your understanding, consider one of the following exploratory tasks:

1. **Use a Pretrained Model**  
   - Experiment with classifying the dataset using a pretrained CNN model (e.g., ResNet, VGG).

2. **Data Augmentation**  
   - Apply data augmentation techniques (e.g., flipping, cropping, rotation) to improve model robustness.

3. **Hyperparameter Tuning**  
   - Experiment with different hyperparameter settings (e.g., learning rates, batch sizes) to optimize performance.

4. **Performance Metrics**  
   - Explore additional performance metrics beyond accuracy (e.g., precision, recall, F1-score).

---

### Notes:
- **Evaluation Criteria**: Each task is assigned points (P) based on its complexity. Ensure your submission includes detailed explanations to get all the points.
- **Deliverables**: Show your scripts, explanations, and any visualizations generated during the lab hand-in session.  
- **Hints**: Read the provided hints carefully; they are meant to guide you through the more challenging aspects of the tasks.
