# Explainable-BERT-based-text-classifier-

## Project Overview

This project addresses the complex task of **Conditional Rule Violation Detection**. Unlike standard toxicity filters that look for "bad words," this model evaluates a Reddit comment based on a specific provided rule and subreddit context.
By utilizing **BERT-Tiny** as a backbone and implementing custom architectural "heads," we achieve high precision while providing **explainability**—allowing us to see why a comment was flagged.

---

## Model Architectures

We developed three distinct classification heads to compare different methods of feature extraction:

| Head Type | Mechanism | Strength |
| --- | --- | --- |
| **Standard BERT** | `[CLS]` Token | Captures a general semantic summary of the entire input. |
| **CNN Head** | 1D-Convolution (Filters: 4, 6, 8, 10) | Identifies **local n-grams** and specific phrase-based violations. |
| **Attention Head** | Trainable Weighted Pooling | Acts as a **"Detective,"** highlighting specific keywords across the text. |

---

## Standard BERT vs CNN Head Model High-Level Architectural Schematic:
<img width="1012" height="313" alt="image" src="https://github.com/user-attachments/assets/75cf62ee-e5eb-41ca-a053-ed82a8e6647b" />
<img width="1024" height="290" alt="image" src="https://github.com/user-attachments/assets/346cd580-09e6-4d77-8e41-0f4145efafb8" />

The first diagram illustrates the data flow from the transformer backbone to the standard head. Use the <CLS> embedding. 
The second diagram illustrates the data flow from the transformer backbone to the convolutional head. We extract the Last Hidden State from BERT, which provides a rich, contextualized embedding for every token in the sequence.

---

## Explainability: Visualizing the Model's Logic
A core priority of this project is moving beyond "Black Box" predictions. By leveraging custom heads on top of BERT, we can audit the decision-making process to understand exactly why a comment was flagged. Comprehensive examples of these visualizations and the code to generate them are included directly in the provided notebook.


CNN Head: Detecting Most Influential Word Combinations
The CNN head is designed to identify specific combinations of words (n-grams) that characterize a rule violation.

Attention Head: Mapping Most Influential Words
The Attention head acts as a "detective," assigning a unique weight of importance to every single token in the input. This allows us to see which individual words across the text were most significant to the final decision.


## Data & Preprocessing

To handle the complexity of rule-based classification, the data was structured into a triple-input format:
`Rule [SEP] Subreddit [SEP] Comment`

* **Dataset Size:** Scaled from 1,600 to **10,000 samples** using Easy Data Augmentation (EDA).
* **Sequence Length:** Optimized at **128 tokens** to balance context and computational efficiency.


## Training Strategy: Two-Stage Fine-Tuning

We implemented a specialized training loop to solve the "CNN Bottleneck" and prevent BERT from "scrambling" its pre-trained knowledge:

1. **Stage 1: Head Warmup (Frozen BERT)**
* BERT is frozen.
* Head trained with a higher Learning Rate ().


2. **Stage 2: Full Fine-Tuning (Unfrozen BERT)**
* BERT is unfrozen.
* Global Learning Rate reduced to  with **Gradient Clipping** to ensure stability.


##  Key Features

* **Modular Design:** Easily switch between CNN, Attention, and Standard heads.
* **Error Analysis:** Includes scripts to isolate False Positives and visualize model "confusion."
* **Lightweight:** Based on BERT-Tiny, making it fast enough to run on standard consumer GPUs or Google Colab.

