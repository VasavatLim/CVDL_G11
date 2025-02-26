# 02 Object Detection Lab

## Introduction
In this lab, you will build a simplified version of the Faster R-CNN model using the Oxford Pets dataset. You will explore object detection architectures—from R-CNN and Fast-RCNN to Faster-RCNN—while diving into data handling, feature extraction, the Region Proposal Network (RPN), and classification.


## Task I: Understand the Model Architecture (1P)
Your first task is to analyze the architecture and understand the evolution of region-based object detection models. Focus on the following points:

1. **Compare the Models:**
    - Describe the differences between R-CNN, Fast-RCNN, and Faster-RCNN.
    - Explain how Faster-RCNN improves upon Fast-RCNN.

2. **Feature Extraction Networks:**
    - Identify the neural network architectures used for feature extraction.
    - Explain the process of extracting feature maps in each of the models (R-CNN, Fast-RCNN, and Faster-RCNN).


## Task II: Enhance Feature Extraction (2P)
This task focuses on improving the feature extraction component of the model:

1. **Properties of Effective Features:**
    - Discuss what makes a feature representation robust and descriptive for object detection tasks.

2. **Explore the VGG Networks:**
    - Define the VGG family of networks.
    - Explain what tasks they were originally designed for and what dataset they were trained on.

3. **Model Modification:**
    - Modify the `FeatureExtractionNetwork` in `model.py` to incorporate a pretrained VGG16 from torchvision.
    - **Hint:** Consider why it is beneficial to freeze the pretrained weights.
    - **Hint:** Identify which part of the VGG16 network will serve as your feature extractor (you need only the first few layers).
    - **Hint:** Determine the output shape of the extracted features.


## Task III: Develop the Region Proposal Network (RPN) (2P)
Transition your RPN from producing random proposals to performing precise bounding box regression:

1. **Network Update:**
    - Modify the RPN to predict bounding box coordinates.
    - **Hint:** Replace the random bbox proposals with a FFNN.

2. **Loss Function:**
    - Explain the rationale behind using the smoothL1Loss in bounding box regression.
    - Pay attention to how the feature extraction output shape factors into the RPN architecture.

3. **Training and Evaluation:**
    - Train the RPN on the bounding box regression task.
    - Implement a performance metric (e.g., MSE, MAE, or IoU) to track improvements and save your trained model.

## Task IV: Visualize Region Proposal Network Predictions (1P)
Visualize the boundingboxes produces by your RPN:

1. **Visualizations script:**
    - implement the main and plot functions in `visualize_rpn.py`


## Task V: Build and Train the Classifier (2P)
Develop a classifier that distinguishes between cats and dogs using your model’s components:

1. **Classifier Design:**
    - Define the classifier’s input parameters and understand how it integrates with the overall Faster R-CNN pipeline.

2. **Training Loop:**
    - Create a training loop that brings together your **pretrained** feature extractor, updated RPN, and the classifier.
    - Train the combined model and evaluate its accuracy on differentiating between cats and dogs.


## Task VI: (Optional) Further Exploration (2P)
For further investigation and to deepen your understanding, consider one of the following exploratory tasks:

1. **Architectural Experiments:**
    - Test alternative architectures for both the feature extraction network and the classifier.

2. **Comparative Analysis:**
    - Compare the performance of your model to a pretrained object detection model (e.g., YOLO from torchvision).
    - Analyze the differences in terms of speed, accuracy, and model complexity.

---

### Notes:
- **Evaluation Criteria**: Each task is assigned points (P) based on its complexity. Ensure your submission includes detailed explanations to get all the points.
- **Deliverables**: Show your scripts, explanations, and any visualizations generated during the lab hand-in session.
- **Hints**: Read the provided hints carefully; they are meant to guide you through the more challenging aspects of the tasks.
