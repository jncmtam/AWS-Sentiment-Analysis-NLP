# Sentiment Analysis Model Report

This report details the training and evaluation process of a sentiment analysis model built using the DistilBERT architecture (`distilbert-base-uncased`). The model classifies text reviews into two categories: **Negative** (label 0) and **Positive** (label 1). Below, we outline the training process, evaluation metrics, and their implications.

### 1. `Dataset Overview`

The dataset consists of **2000 samples**, split as follows:

- **Negative (label 0)**: 1035 samples
- **Positive (label 1)**: 965 samples

A test set of **400 samples** (199 Negative and 201 Positive) was used for evaluation.

### 2. `Training Process`

The model was initialized with the `distilbert-base-uncased` checkpoint. Some weights (e.g., `classifier.bias`, `classifier.weight`, `pre_classifier.bias`, `pre_classifier.weight`) were not pre-initialized and were trained from scratch for the sentiment classification task. Training occurred over **2 epochs**, with the following loss values:

- **Epoch 1**:
  - **Train Loss**: 0.4362
  - **Validation Loss**: 0.2271
- **Epoch 2**:
  - **Train Loss**: 0.1970
  - **Validation Loss**: 0.2129

### 3. `Observations`

- The **train loss** decreased significantly from 0.4362 to 0.1970, indicating effective learning on the training data.
- The **validation loss** dropped from 0.2271 to 0.2129, suggesting good generalization to unseen data.
- The small gap between train and validation loss implies minimal overfitting.

### 4. `Evaluation Metrics`

The model’s performance was assessed on the test set, yielding the following results:

- **Accuracy**: 0.8775 (87.75%)

### 5. `Classification Report`

Detailed metrics for each class are presented below:

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Negative         | 0.85      | 0.91   | 0.88     | 199     |
| Positive         | 0.90      | 0.85   | 0.87     | 201     |
| **Macro Avg**    | 0.88      | 0.88   | 0.88     | 400     |
| **Weighted Avg** | 0.88      | 0.88   | 0.88     | 400     |

#### Metric Definitions

- **Precision**: Proportion of correct predictions among all predictions for a class.
  - Negative: 85% of predicted Negative reviews were correct.
  - Positive: 90% of predicted Positive reviews were correct.
- **Recall**: Proportion of actual instances of a class correctly identified.
  - Negative: 91% of actual Negative reviews were detected.
  - Positive: 85% of actual Positive reviews were detected.
- **F1-Score**: Harmonic mean of precision and recall, balancing the two metrics.
  - Negative: 0.88
  - Positive: 0.87
- **Support**: Number of samples per class in the test set.

### 6. `Interpretation`

- **Accuracy (87.75%)**: The model correctly classified 87.75% of test samples, a strong result for binary classification.
- **Negative Class**: Higher recall (0.91) indicates the model excels at identifying Negative reviews, though precision (0.85) suggests some false positives.
- **Positive Class**: Higher precision (0.90) shows fewer false positives, but recall (0.85) indicates some Positive reviews were missed.
- **Balanced Performance**: Near-identical F1-scores (0.88 and 0.87) and macro/weighted averages (0.88) demonstrate consistent performance across classes, with no significant bias.

#### What These Parameters Indicate

- **Effective Learning**: Decreasing loss values over epochs show the model successfully learned sentiment patterns.
- **Good Generalization**: Low validation loss and high test accuracy suggest the model performs well on new data.
- **Class Balance**: The model handles both Negative and Positive reviews effectively, with slight trade-offs between precision and recall.
- **Practical Utility**: An accuracy of 87.75% and F1-scores near 0.88 make this model reliable for real-world sentiment analysis tasks.

### 7. `Conclusion`

The DistilBERT-based sentiment analysis model achieves strong performance with an accuracy of 87.75% and balanced metrics across classes. The training process indicates effective learning and generalization, making it a solid foundation for sentiment classification applications.

### Author : `Chu Minh Tam` - ML/NLP Researcher

### Contact :

- jn.cmtam@gmail.com
- tam.chu2213009cs@hcmut.edu.vn
- +84 327628468

