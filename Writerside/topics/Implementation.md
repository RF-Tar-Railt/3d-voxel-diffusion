# 4. Implementation

This section details the two core implementation approaches—traditional machine learning (SVM) and deep learning (BERT)—including model architecture, experimental environment configuration, and the challenges encountered along with their solutions during training.

## 4.1 Model Implementation

### 4.1.1 SVM Method Implementation

Support Vector Machine (SVM), as a classical and efficient machine learning algorithm, has demonstrated excellent performance in text classification tasks **[7]**. In this study, the main steps for the SVM implementation are as follows:

1. **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) is used to extract text features. 5000 of the most representative features are selected to convert unstructured medical record texts into vector representations that can be processed by machine learning algorithms.
   
2. **Data Splitting**: The dataset consists of 800 training samples and 200 test samples, maintaining the original distribution.

3. **Model Training**: The SVM model is implemented using the sklearn library, with a linear kernel to balance computational efficiency and classification performance.

Due to its high computational efficiency and strong performance in small-sample learning, the SVM method serves as the baseline model in this study, providing a reference standard for evaluating more complex models.

### 4.1.2 Pre-trained BERT Method Implementation

BERT (Bidirectional Encoder Representations from Transformers) is a breakthrough technology in natural language processing in recent years **[2]**. Its bidirectional encoding mechanism can capture semantic information from texts more comprehensively. In this study, task-specific fine-tuning is performed on the Chinese pre-trained model bert-base-chinese, as described below:

1. **Model Architecture Design**: A combined architecture of shared parameters and task-specific parameters is used. As illustrated, the first 6 layers of the BERT model share parameters between two classification tasks. This is based on the assumption that lower-level feature extraction is generic across different classification tasks; the subsequent 6 layers are separately trained for the two distinct classification tasks (single-label main diagnosis classification and multi-label other diagnosis classification) to capture task-specific features.

![BERT结构说明.png](BERT结构说明.png)

2. **Data Splitting**: The training set contains 720 samples, the validation set 80 samples (a 9:1 split), and the test set 200 samples. This split ensures that the generalization ability of the model is continuously evaluated during training.

## 4.2 Experimental Environment Configuration

To ensure the reproducibility of the study, the software and hardware environments are documented in detail:

- **Computing Platform**: Google Colab Pro
- **Computing Resources**: NVIDIA T4 GPU (15GB VRAM)
- **Data Preprocessing Framework**:
  - paddlepaddle-gpu 2.6.0
  - paddlenlp 2.8.1
- **Model Implementation Framework**:
  - SVM: scikit-learn
  - BERT: PyTorch and Transformers library

## 4.3 Implementation Challenges and Solutions

The primary challenge during the experiments was the imbalanced distribution of diagnostic codes, especially the extremely imbalanced distribution of other diagnosis codes. This imbalance causes the model to focus on learning the main diagnostic code while neglecting the minority classes in other diagnosis codes. To address this, two complementary methods were adopted:

### 4.3.1 Category Weight Adjustment

To balance the impact of different categories, a category weighting mechanism is introduced. For the multi-label classification task (other diagnosis), inverse proportional weights are calculated based on the frequency of occurrence of each category in the training set, ranging from 1 to 500. This ensures that the model pays more attention to rare categories during training, as illustrated below:

- For each category:
  - `total_samples - label2_counts` represents the number of negative samples (samples not belonging to the category).
  - `label2_counts` represents the number of positive samples.
  - `1e-6` is a small constant (ε) to prevent division by zero (in cases where no positive samples exist for a category).

The formula is as follows:

$$
\text{pos weights}[j] = \frac{\text{total samples} - \text{label2 counts}[j]}{\text{label2 counts}[j] + 10^{-6}}
$$

This method effectively improves the recognition of low-frequency classes, thereby enhancing overall classification performance.

### 4.3.2 Uncertainty Weighted Loss Function

To balance the single-label and multi-label classification tasks, the uncertainty weighted loss function proposed by Kendall et al. **[8]** is implemented. This method allows the model to automatically learn the optimal weight for each task without manual tuning.

The implementation is as follows:

1. **Loss Function for Single-label Classification (Cross-Entropy Loss)**:

$$
L_1 = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{p}_{i,c})
$$

#### Parameter Description: {id="parameter-description_1"}
- **N**: Total number of samples.
- **C**: Total number of classes.
- **y_{i,c}**: The true label for sample i in class c (usually represented in one-hot encoding).
- **$\hat{p}_{i,c}$**: The predicted probability for sample i in class c, calculated as:
  $$
  \hat{p}_{i,c} = \text{softmax}(\text{logits1})_{i,c}
  $$

#### Note: {id="note_1"}
The `softmax` function converts logits into a probability distribution, ensuring that **$\hat{p}_{i,c}$** falls within the range [0, 1].

2. **Loss Function for Multi-label Classification (Weighted Binary Cross-Entropy Loss)**:

$$
L_2 = -\frac{1}{N} \sum_{i=1}^{N} \left[ w_p \cdot y_i \cdot \log(\hat{p}_i) + (1 - y_i) \cdot \log(1 - \hat{p}_i) \right]
$$

#### Parameter Description: {id="parameter-description_2"}
- **N**: Number of samples.
- **y_i**: The true label of the i-th sample (0 or 1).
- **$\hat{p}_i$**: The predicted probability, computed as:
  $$
  \hat{p}_i = \text{sigmoid}(\text{logits2}_i)
  $$
- **$w_p$**: The positive class weight, used to adjust the imbalance between positive and negative samples.

#### Note:
This formula is suited for binary classification problems in imbalanced datasets, adjusting the importance between positive and negative samples through the weight $w_p$.

3. **Combined Loss Function**:

$$
L_{total} = \frac{L_1}{2\sigma_1^2} + \frac{L_2}{2\sigma_2^2} + \log(\sigma_1^2) + \log(\sigma_2^2)
$$

#### Parameter Description:
- **L_1** and **L_2** are the two components of the loss function.
- **$\sigma_1^2$** and **$\sigma_2^2$** represent the variances of the loss components.
- **$\log(\sigma_1^2)$** and **$\log(\sigma_2^2)$** are logarithmic terms used to regularize or manage the variances.

Here, we set $\sigma_1$ as -1 for single-label classification and $\sigma_2$ as 1 for multi-label classification as the initial values. During training, these parameters are automatically adjusted via gradient descent, achieving a dynamic balance between the tasks.

The introduction of this uncertainty weighted loss function significantly enhances the joint optimization of both tasks, improving the overall performance of the model on both classification tasks.

### 4.3.3 Front-end System Implementation and Challenges

The front-end system consists of the following key components:
1. **Model Initialization and Loading**: Responsible for loading the pre-trained BERT model and its parameters.
2. **Test Data Loading**: Reads the test dataset from a JSON file.
3. **Medical Record Selection and Display**: Allows users to browse and select specific medical record cases.
4. **Prediction Function**: Processes user inputs and generates corresponding diagnostic code predictions.

![frontEnd](frontEnd.jpg)

During the implementation of the front-end system, we encountered the following technical challenges along with corresponding solutions:

1. **Model Loading Performance**
   - **Challenge**: Initializing and loading the large BERT model during page load causes slow loading times.
   - **Solution**: Implement a lazy loading strategy where model inference is executed only after the user clicks the "Predict" button, with a loading indicator added to enhance user experience.

2. **Handling Long Text**
   - **Challenge**: Medical record texts are typically long and may exceed the maximum input length of the BERT model (512 tokens).
   - **Solution**: Configure the tokenizer with truncation=True to automatically truncate overly long texts, preserving the most important beginning and ending segments to retain key information.

3. **Multi-label Classification Display**
   - **Challenge**: The "other diagnosis" task is a multi-label classification problem that requires displaying multiple prediction results simultaneously.
   - **Solution**: Format the multi-label prediction results as a list for ease of viewing, and utilize color coding to distinguish between different predictions.

4. **User Interface Adaptation**
   - **Challenge**: The lengthy medical record texts and prediction results may not display consistently across different screen sizes.
   - **Solution**: Use Streamlit's responsive layout features and add scrollbars for long texts to ensure optimal display on various devices.

These solutions significantly improved the usability and user experience of the system, enabling medical professionals to use the tool more effectively for ICD diagnostic code assistance.
