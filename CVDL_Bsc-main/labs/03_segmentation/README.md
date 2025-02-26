# 03 Semantic Segmentation Lab

## Introduction
In this lab, you will build a semantic segmentation model using the Oxford Pets dataset. Your goal is to implement and experiment with U-Net architectures, including both the classic and attention variants, to accurately segment objects in these images.


## Task I: Understand the U-Net Architecture (1P)
Examine the provided `model.py` file and review the [original U-Net paper](https://arxiv.org/abs/1505.04597) to deepen your understanding of segmentation models. Address these points:

1. **Encoding Path:**
   - Describe the purpose of the encoding (contracting) path and how it captures image context.

2. **Skip Connections:**
   - Explain why skip connections are used in U-Net and how they assist in recovering spatial information during decoding.

3. **Building Blocks:**
   - Evaluate the building blocks provided in `model.py`. Are they sufficient to construct a complete U-Net? Identify any missing components.


## Task II: Implement Segmentation Metrics (1P)
Enhance your training script `train.py` by adding evaluation metrics critical for segmentation:

1. **Intersection over Union (IoU):**
   - Implement the IoU metric to assess segmentation accuracy.
   - *Hint:* Examine the mask dimensions in the dataset to ensure proper metric implementation.


## Task III: Implement the Classic U-Net Model (2P)
Expand the U-Net model in `model.py` by implementing the classic design as described in the original paper:

1. **Model Construction:**
   - Assemble the U-Net architecture using the provided components, ensuring that feature maps are handled correctly.
   - *Hint:* Monitor the sizes of the feature maps throughout the network.

2. **Data Transformations:**
   - Include any necessary cropping or resizing transformations so that input and output sizes properly align.

3. **Output Channels:**
   - Determine the correct number of output channels. For a segmentation task with three classes (foreground, background, unknown), ensure your final layer reflects this.


## Task IV: Implement the Attention U-Net (2P)
Create an alternative model variant by incorporating attention mechanisms in your U-Net implementation:

1. **Model Variation:**
   - Add a new model in `model.py` that implements the attention U-Net architecture.

2. **Key Considerations:**
   - Verify that consistent padding is maintained throughout the network.
   - *Hint:* Identify the activation functions used and explain why they are appropriate in this context.


## Task V: Train and Compare Models (2P)
Evaluate the performance of your classic and attention U-Net implementations by training and comparing them:

1. **Training Scripts:**
   - Develop separate training scripts for each network variant.
   - *Hint:* Don’t forget to save your trained models.

2. **Result Visualization:**
   - Create plots to display the original image, the ground truth mask, the prediction from the classic U-Net, and the prediction from the attention U-Net.

3. **Performance Analysis:**
   - Use the IoU metric to compare the models quantitatively.


## Task VI: (Optional) Further Exploration Ideas (2P)
For further investigation and to deepen your understanding, consider one of the following exploratory tasks:

1. **Additional Architectures:**
   - Implement another segmentation model, such as U-Net++.

2. **Comparative Analysis:**
   - Compare your U-Net models against a Masked R-CNN for segmentation tasks.

3. **Post-Processing Techniques:**
   - Apply post-processing methods to refine your segmentation outputs.

4. **Ensemble Strategies:**
   - Experiment with ensemble techniques to potentially boost segmentation performance.

---

### Notes:
- **Evaluation Criteria**: Each task is assigned points (P) based on its complexity. Ensure your submission includes detailed explanations to get all the points.
- **Deliverables**: Show your scripts, explanations, and any visualizations generated during the lab hand-in session.
- **Hints**: Read the provided hints carefully; they are meant to guide you through the more challenging aspects of the tasks.
