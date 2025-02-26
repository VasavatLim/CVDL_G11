# 04 Vision Transformers Lab

## Introduction
In this lab, you will explore Vision Transformers (ViTs), a type of model that applies transformer architectures—originally designed for natural language processing—to computer vision tasks. You will implement your own ViT model, train it, fine-tune a pretrained ViT, and compare its performance to hierarchical transformers and convolutional neural networks (CNNs).


## Task I: Understand the Vision Transformer Architecture (1P)
Your first task is to study the concept and architecture of Vision Transformers by reviewing the [original ViT paper](https://arxiv.org/abs/2010.11929) and the foundational [Transformer paper](https://arxiv.org/abs/1706.03762). Focus on the following:

1. **Key Differences:**
   - Summarize how Vision Transformers differ from traditional CNNs in processing images.

2. **Self-Attention Mechanisms:**
   - Explain the role of self-attention mechanisms in capturing global dependencies within an image.


## Task II: Implement a Vision Transformer (1P)
In this task, you will complete the partial implementation of a ViT model provided in `model.py`. Follow these steps:

1. **Patch Embedding:**
   - Complete the `PatchEmbed` module by addressing the `TODOs`.
   - **Hint:** Use `Conv2D` or `torch.unfold` to divide images into patches.

2. **MLP Module:**
   - Complete the `MLP` module by addressing the `TODOs`.
   - **Hint:** Refer to the activation function mentioned in the original ViT paper.

3. **Attention Mechanism:**
   - Complete the `Attention` module by addressing the `TODOs`.
   - **Hint:** Use integer division (`//`) where necessary.


## Task III: Train a Vision Transformer from Scratch (2P)
Once your model is implemented, train it from scratch using the provided `train.py` script.

1. **Training Setup:**
   - Configure the parameters at the top of `train.py` appropriately.
   - Ensure that image dimensions are divisible into patches.

2. **Experiment with Hyperparameters:**
   - Try different hyperparameter configurations (e.g., learning rate, batch size) and log your results.
   - **Hint:** Don't expect high performance when training from scratch; explain why results may be suboptimal.

3. **Save Your Model:**
   - Save your trained model for further comparison.


## Task IV: Fine-Tune a Pretrained Vision Transformer (2P)
In this task, you will fine-tune a pretrained ViT model using `finetune.py`. The pretrained model was trained on ImageNet21K.

1. **Adapt Output Layer:**
   - Modify the output layer to match your dataset's number of classes.

2. **Fine-Tuning Approaches:**
   - Fine-tune the entire pretrained model on your dataset and log its performance.
   - Fine-tune only the last layer of the pretrained model while freezing all other weights. Log your results for this approach as well.

3. **Comparison:**
   - Compare the performance of:
     - The model trained from scratch (Task III).
     - The fully fine-tuned pretrained model.
     - The pretrained model with only its last layer fine-tuned.


## Task V: Compare with Hierarchical Vision Transformers (2P)
Explore hierarchical transformer models like Swin Transformers by reviewing their model card in [torch-image-models](https://huggingface.co/timm/swinv2_base_window12to16_192to256.ms_in22k_ft_in1k) and their [original publication](https://arxiv.org/abs/2103.14030).

1. **Key Differences:**
   - Explain how hierarchical transformers differ from standard ViTs in architecture and functionality.

2. **Fine-Tuning a Swin Transformer:**
   - Fine-tune a Swin Transformer on the Oxford Pets dataset and log its performance.

3. **Comparison:**
   - Compare your results with those obtained from Task IV (fine-tuned ViTs).


## Task VI: Advanced Exploration (Optional) (2P)
For further investigation and to deepen your understanding, consider one of the following exploratory tasks:

1. **CNN Comparison:**
   - Implement a CNN with approximately the same number of parameters as your ViT and compare their performance on your dataset.

2. **Explore More Architectures:**
   - Experiment with additional architectures available on [pytorch-image-models](https://huggingface.co/timm).

3. **Model Sharing:**
   - Publish your trained model on Hugging Face to share it with classmates and version it for future use.


---

### Notes:
- **Evaluation Criteria**: Each task is assigned points (P) based on its complexity. Ensure your submission includes detailed explanations to get all the points.
- **Deliverables**: Show your scripts, explanations, and any visualizations generated during the lab hand-in session.
- **Hints**: Read the provided hints carefully; they are meant to guide you through the more challenging aspects of the tasks.
