# 01 Image Classification Lab

## Introduction
In this lab, you will build and experiment with an image classification model using the [Oxford Pets dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). This dataset consists of images of varying resolutions, each depicting either a dog or a cat. You will explore neural network architectures, train models, analyze their performance, and gain insights into how Convolutional Neural Networks (CNNs) process image features.


## Task I: Understand the Provided Python Scripts (1P)
Your first task is to analyze the provided Python scripts to understand how the initial model works. Focus on the following key aspects:

1. **Network Architecture**  
   - Identify and describe the layers of the neural network.
   
2. **Optimizer**  
   - Explain the choice of the optimizer and its role in training the model.

3. **Loss Function**  
   - Review the loss function used and explain why it is appropriate for this classification task.

4. **TensorBoard Logging**  
   - Examine how TensorBoard is integrated into the training process.  
   - Describe the benefits of using TensorBoard for monitoring training progress.


## Task II: Familiarize Yourself with the Dataset (1P)
The Oxford Pets dataset will serve as the foundation for model training. In this task, you will explore it in detail:

1. **Dataset Overview**  
   - Analyze the variety of images in the dataset (e.g., resolution, composition, etc.).  
   - Describe the types of information you can extract from the dataset (e.g., labels, image dimensions, etc.).

2. **Insights Gained**  
   - Provide a summary of your observations about the dataset's structure and content.


## Task III: Implement a CNN (2P)
Your goal in this task is to design and implement a Convolutional Neural Network (CNN) with dense (fully connected) layers for image classification. This involves the following steps:

1. **Build the Model**  
    - Specify the layers of your CNN, such as convolutional layers, pooling layers, and dense layers.  
    - Clearly explain why dense layers are necessary for the classification process.

2. **Prepare the Data**
    - Resize the images to ensure uniform input dimensions.  
    - **Hint:** Calculate the average height and width of the images in the dataset to choose appropriate resize parameters.  


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
