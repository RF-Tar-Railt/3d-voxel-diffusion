# 3 Methodology

This study aims to use machine learning methods, particularly deep learning models, for performing ICD diagnostic coding on Chinese electronic medical records. This section details the methodology of this study, including the research design, data source and preprocessing, model selection and construction, experimental setup, and evaluation metrics.

## 3.1 Research Design

In this study, a supervised learning approach is adopted to build a multi-class model for predicting ICD diagnostic codes from Chinese electronic medical records. Specifically, the electronic medical records are used as input and the ICD codes as output, training the model to learn the mapping between texts and codes. For comparison, Support Vector Machine (SVM) is also introduced as a traditional machine learning method.

Below is the flowchart of the implemented method:

![main_method](主要流程.png)

## 3.2 Data Source and Preprocessing

### 3.2.1 Data Source

The dataset used in this study comes from the 2025 Tianchi Competition on ICD diagnostic coding of Chinese electronic medical records. The data has been anonymized. The training set contains 800 cases, and the test set contains 200 cases. Each case is a JSON file containing 14 fields with the following contents:

-   **Medical Record ID**: The unique identifier of the patient's medical record during hospitalization.
-   **Chief Complaint**: The primary reason for the patient’s visit.
-   **Present Illness History**: A detailed description of the onset, development, and evolution of the current illness.
-   **Past Medical History**: The patient’s previous health status and history of diseases.
-   **Personal History**: Information on lifestyle habits, occupational exposures, contact with epidemic areas, etc.
-   **Marital History**: Details of marital status, age at marriage, spouse’s health condition, whether the patient has children, and sexual history.
-   **Family History**: The existence of hereditary diseases or specific conditions in immediate family members.
-   **Admission Condition**: Symptoms, signs, and general condition upon admission.
-   **Admission Diagnosis**: The preliminary diagnosis based on the current illness history and examinations.
-   **Diagnosis and Treatment Process**: Detailed records of examinations, treatments, and changes in condition during hospitalization.
-   **Discharge Condition**: A brief description of the patient’s health condition and recovery at discharge.
-   **Discharge Instructions**: Guidelines for lifestyle, medication, and follow-up after discharge.
-   **Primary Diagnosis Code**: The standardized code corresponding to the primary diagnosis during hospitalization.
-   **Other Diagnosis Codes**: The standardized codes corresponding to other relevant diagnoses during hospitalization.

Statistics are provided for both the primary diagnosis code and the other diagnosis codes in the training set.

The distribution of the primary diagnosis codes is relatively uniform:

![primary_dc](primary_diagnosis_codes.png)

In contrast, the distribution of the other diagnosis codes varies significantly:

![other_dc](other_diagnosis_codes.png)

The table below shows the correspondence between admission diagnosis and diagnosis codes in the text data.

|                     Field                      |                     Content                     |
|:----------------------------------------------:|:-----------------------------------------------:|
|              Admission Diagnosis               | 1.冠状动脉粥样硬化性心脏病不稳定型心绞痛2.高血压病（3级很高危）3.高脂血症4.甲状腺术后 |
| Primary Diagnosis Code & Other Diagnosis Codes |    I20.800x007 & I10.x00x032;E11.900;E78.500    |

### 3.2.2 Data Preprocessing

Since the raw electronic medical record texts contain a large amount of unstructured information, preprocessing is required before training the model. The following preprocessing steps are adopted:

1.  **Text Merging**: Merge the texts from Chief Complaint, Present Illness History, Admission Condition, the first text in Admission Diagnosis, Diagnosis and Treatment Process, and Discharge Condition into the primary diagnosis text. Merge the texts from Past Medical History, Personal History, Marital History, Family History, the texts in Admission Diagnosis except the first, and Discharge Instructions into the other diagnosis text.
2.  **Entity and Relation Extraction**: Utilize the entity and relation extraction functionalities in PaddleNLP for medical text to extract entities such as diseases, symptoms, examination items, treatment methods, body parts, onset time, medications, and surgeries. **[6]**

After preprocessing, the electronic medical record texts are converted into structured data that can be better utilized for model training.

*Example of before and after preprocessing*

Before merging texts:

![处理前.jpg](处理前.jpg)

After processing:

![处理后.jpg](处理后.jpg)

## 3.3 Model Selection and Construction

### 3.3.1 Model Selection

This study selects the BERT model as the base model. BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained deep learning model that has achieved state-of-the-art performance in natural language processing **[2]**. BERT leverages the Transformer **[5]** encoder architecture to capture contextual information in the text, thereby better understanding the semantics.

Additionally, to validate the advantages of deep learning models, Support Vector Machine (SVM) is employed as a baseline method. SVM is a classic supervised learning algorithm which performs well in text classification tasks. By mapping text features into a high-dimensional space, SVM can find the optimal classification hyperplane for text categorization. In this study, TF-IDF (Term Frequency-Inverse Document Frequency) is used to extract features, converting texts into vector representations which are then used to train the SVM model.

### 3.3.2 Model Construction

The study uses a pre-trained BERT model with an added fully connected layer and a Softmax layer for classifying ICD diagnostic codes. The model architecture is as follows:

1.  **Embedding Layer**: Converts input texts to word embeddings.
2.  **Positional Encoding Layer**: Adds positional information to the word embeddings.
3.  **Transformer Encoder Layer**: Consists of 12 encoder layers to capture contextual information in the text.
4.  **Fully Connected Layer**: Maps the output from the Transformer encoder to the dimensions of the ICD diagnostic codes.
5.  **Softmax Layer**: Calculates the probability for each ICD diagnostic code.

![BERT.png](BERT.png)

The Transformer Encoder Layer consists of the following modules:

-   **Multi-Head Attention**: Multi-head attention mechanism that uses multiple sets of q, k, v matrices to compute attention scores and extract information.
-   **Feed Forward**: A fully connected feed-forward neural network.
-   **Add & Norm**: A combination of residual connection and layer normalization: the input is added to the output to form a new output which is then normalized.

### 3.3.3 Technology Stack and Tools

The frontend of this project is implemented using the Streamlit framework, which is particularly useful for machine learning and data science projects as it enables rapid development of interactive web applications. The technology stack includes:

- **Streamlit**: Used to build interactive user interfaces.
- **PyTorch**: Deep learning framework for loading and running the pre-trained model.
- **Transformers**: Hugging Face's library for loading pre-trained BERT models and tokenizers.
- **JSON**: Used for data exchange and storing test data.

### 3.3.4 System Design and Architecture

The system follows a Single Page Application (SPA) design philosophy and is divided into three main functional areas:

1. **Model Loading Area**: Responsible for initializing the model and loading pre-trained parameters.
2. **Data Display Area**: Displays the list of test medical records and allows users to select a specific record.
3. **Prediction Results Area**: Shows the predicted diagnostic results for the selected medical record.

Data processing in the frontend involves the following steps:

1. **Loading Test Data**: Load the test dataset from JSON files.
2. **Text Preprocessing**: Use the BERT tokenizer to segment and encode patient record texts, including:
   - Truncating or padding texts to a fixed length (512).
   - Adding special tokens ([CLS], [SEP]).
   - Generating attention masks and token type IDs.
3. **Post-processing Predictions**: Convert the model's output probabilities into diagnostic codes.

## 3.4 Evaluation Metrics

For the ICD diagnostic coding task on Chinese electronic medical records, this study uses Accuracy (Acc) as the evaluation metric, calculated as follows:

$$
Acc = \frac{1}{N} \sum_{i=1}^{N} \left\{ 0.5 \cdot I(\hat{y}_{main} == y_{main}) + 0.5 \cdot \frac{NUM(y_{other} \cap \hat{y}_{other})}{NUM(y_{other})} \right\}_{i}
$$

Here, $I(\cdot)$ is an indicator function which returns 1 if the condition is met, and 0 otherwise; $\hat{y}_{main}$ and $y_{main}$ represent the predicted and true labels for the primary diagnosis respectively; $NUM(x)$ denotes a function that counts the number of elements in $x$; $\hat{y}_{other}$ and $y_{other}$ represent the sets of predicted and true labels for the other diagnoses; $N$ is the total number of test samples, and $\left\{\cdot\right\}_{i}$ denotes the prediction accuracy for the i-th electronic medical record.
