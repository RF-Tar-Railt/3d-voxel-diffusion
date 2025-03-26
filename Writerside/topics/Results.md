# 5. Results

This section presents the experimental results in detail and provides an in-depth analysis and discussion. We first display the overall performance of the two methods, then analyze the training process and performance characteristics of the pre-trained BERT model.

## 5.1 Overall Experimental Results

Table 5.1 shows the classification accuracy of two methods on the ICD diagnosis coding task using Chinese electronic medical records. The results clearly indicate that the deep learning method based on pre-trained BERT significantly outperforms the traditional SVM method, with an improvement of 12.25 percentage points in classification accuracy.

**Table 5.1 Comparison of Experimental Results for Different Methods**

| Method           | Classification Accuracy |
|------------------|-------------------------|
| SVM              | 59.00%                  |
| Pre-trained BERT | 71.25%                  |

The SVM method performs relatively weakly, which may be due to:
1. TF-IDF features not effectively capturing the semantic information and contextual relationships in text;
2. The specialized terminology and complex expressions in medical texts making a simple bag-of-words model inadequate for accurate representation;
3. The inherent limitations of linear SVM in handling multi-class classification problems.

In contrast, the pre-trained BERT method significantly improves classification performance, mainly because:
1. The BERT model acquires abundant language knowledge through pre-training;
2. The bidirectional attention mechanism better captures the contextual information in medical texts;
3. The architecture that combines shared and task-specific parameters effectively handles the differences between single-label and multi-label classification tasks;
4. A specially designed loss function is applied to address data imbalance.

## 5.2 Analysis of the Pre-trained BERT Model Training Process

To gain a deeper understanding of the training process and performance evolution of the pre-trained BERT model, key indicators during training have been visualized.

### 5.2.1 Loss Trend Analysis

Figure 5.1 illustrates the trend of loss values for both the training and validation sets during the training process.

![loss_curve.png](loss_curve.png)

**Figure 5.1 Loss Curves for the Training and Validation Sets**

The loss curves reveal the following key observations:
1. **Convergence Speed**: The model converges quickly in the initial epochs with a rapid drop in loss.
2. **Training Stability**: After the 5th epoch, the training loss continues to decrease steadily while the validation loss starts fluctuating, yet overall follows a downward trend.
3. **Onset of Overfitting**: Around the 16th epoch, the validation loss begins to rise, indicating that the model may start overfitting the training data.

Thus, selecting 16 epochs as the number of training rounds is reasonable. It ensures full utilization of the training data (processing 45 samples per epoch) and stops training before significant overfitting occurs, thereby balancing the model's learning ability and generalization performance.

### 5.2.2 Classification Accuracy Analysis

Figure 5.2 displays the accuracy trends for both training and validation sets throughout the training process.

![accuracy_curve.png](accuracy_curve.png)

**Figure 5.2 Classification Accuracy Trends for the Training and Validation Sets**

From the accuracy curves, the following conclusions can be drawn:
1. **Learning Effectiveness**: Both the training and validation accuracies demonstrate an upward trend as training progresses, indicating effective learning of diagnostic coding patterns from the data.
2. **Peak Performance**: The validation accuracy peaks at the 9th epoch and then fluctuates slightly, remaining largely stable without significant further improvement.
3. **Model Selection**: Based on the early stopping strategy [9], the model parameters at the epoch with the highest validation accuracy (the 9th epoch) were saved as the final model, which helps to enhance generalization on unseen data.

The continuous increase in training accuracy coupled with an early plateau of validation accuracy is characteristic of deep learning models, demonstrating that the model has effectively captured the data patterns and possesses good generalization ability.

### 5.2.3 Analysis of Uncertainty Weight Parameters

Figure 5.3 shows the trend of sigma values in the uncertainty-weighted loss function during training.

![sigma_curve.png](sigma_curve.png)

**Figure 5.3 Trend of Sigma Values in the Uncertainty Weighted Loss**

From the sigma value trends, we observe that:
1. **Parameter Stability**: The two sigma values (σ₁ and σ₂) oscillate within a small range during training, indicating that the initial settings (σ₁ = -1, σ₂ = 1) are relatively reasonable and effectively balance the losses of the two tasks.
2. **Task Balance**: The stable sigma values demonstrate that the uncertainty-weighted loss function successfully balances the single-label and multi-label classification tasks, preventing either task from dominating the training process.
3. **Convergence Characteristics**: The sigma values converge in later stages of training, suggesting that the optimal weight distribution between the tasks has been achieved.

This stable behavior of sigma values indicates that our multi-task learning framework has successfully maintained the independence of the two tasks while achieving collaborative optimization, which aligns with the multi-task learning theory proposed by Kendall et al. [8].

