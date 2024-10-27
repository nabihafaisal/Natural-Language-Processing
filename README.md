
# Toxic Comment Classification with BERT

This project performs toxic comment classification using a fine-tuned BERT model. It identifies and categorizes text data into "toxic" or "non-toxic" classes. The model is trained and evaluated on a dataset containing text labeled with toxic and non-toxic comments, using PyTorch and Hugging Face Transformers in a Google Colab environment.

![Confusion Matrix](path_to_your_confusion_matrix_image.png)  
*Confusion Matrix of the Model’s Performance*

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Training and Evaluation](#training-and-evaluation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview

This repository contains the code for a binary text classification project using BERT, aimed at detecting toxic comments. It uses data augmentation techniques to balance the dataset and leverages Hugging Face's `transformers` library for implementing BERT. Evaluation metrics such as accuracy, F1-score, and a confusion matrix are used to measure model performance.

## Dataset

The dataset consists of labeled comments for toxicity detection with two classes:
- **0**: Non-toxic
- **1**: Toxic

Each row in the dataset has a `Text` column containing the comment and a `Toxic` column with the label. The dataset is balanced using the `RandomOverSampler` from the `imblearn` library.

## Model Architecture

We use the `BertForSequenceClassification` model from Hugging Face, specifically `bert-base-uncased`, with two output labels (toxic and non-toxic). The BERT model is fine-tuned on the toxic comment dataset.

## Installation

To run this project, you will need Python 3 and the following libraries:

```bash
pip install torch transformers scikit-learn imbalanced-learn matplotlib
```

If you're using Google Colab, the GPU runtime should be enabled to speed up model training and evaluation.

## Training and Evaluation

### Data Preprocessing

1. **Data Balancing**: We use the `RandomOverSampler` from `imbalanced-learn` to balance the classes.
2. **Data Tokenization**: We tokenize the text data using BERT's `BertTokenizer`, with padding and truncation for sequence length consistency.
3. **Data Splitting**: The balanced data is split into training and testing sets using an 80-20 split.

### Model Training

The model is trained using a custom training loop with the following steps:

- **Epochs**: We iterate over 3 epochs until we observe satisfactory performance.
- **Loss Calculation**: `CrossEntropyLoss` is used as the loss function.
- **Optimizer**: We use the Adam optimizer with a learning rate of 2e-5.

### Evaluation

The `evaluate_model` function computes the following metrics:
- **Accuracy**: Overall accuracy of the model.
- **Classification Report**: Precision, Recall, F1-score for each class.
- **Confusion Matrix**: Visualizes model performance for both toxic and non-toxic labels.

```python
# Example of evaluating the model
accuracy, avg_loss, class_report, conf_matrix = evaluate_model(model, test_loader, device)
```

## Usage

1. Clone this repository and navigate to the project directory.

   ```bash
   git clone https://github.com/yourusername/toxic-comment-classification.git
   cd toxic-comment-classification
   ```

2. Run the Jupyter notebook or Python script to preprocess data, train the model, and evaluate its performance. 

   - **In Google Colab**: Load the notebook and enable the GPU runtime.
   - **In a local environment**: Install required dependencies and execute each cell sequentially.

3. To use the model on new data, load the saved model and tokenizer:

   ```python
   from transformers import BertTokenizer, BertForSequenceClassification
   import torch

   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   model = BertForSequenceClassification.from_pretrained('path_to_saved_model')
   ```

   Then, tokenize new sentences and pass them through the model for prediction.

## Results

The model achieved the following performance on the test set:

- **Accuracy**: 99.2%
- **Average Loss**: 0.035
- **F1-Score**: 0.99 for both toxic and non-toxic classes

The confusion matrix and classification report indicate that the model is effective at distinguishing between toxic and non-toxic comments with minimal false positives and false negatives.

## Confusion Matrix

The confusion matrix below demonstrates the model's performance:

|               | Predicted Non-Toxic | Predicted Toxic |
|---------------|----------------------|-----------------|
| True Non-Toxic| 28241               | 424             |
| True Toxic    | 10                  | 28879           |



## Acknowledgements

- **Hugging Face** for providing pretrained BERT models and tokenizers.
- **scikit-learn** and **imbalanced-learn** for useful utilities in evaluation and data balancing.
- **Google Colab** for accessible cloud GPU support.

---
