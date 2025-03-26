# 1. Introduction

The aging global population and the increasing demand for healthcare services have underscored the importance of 
efficient data management and analysis in medical systems. Electronic Medical Records (EMRs) have emerged as a critical
tool in this regard, facilitating the standardized management and sharing of clinical data. The International 
Classification of Diseases (ICD) system, maintained by the World Health Organization (WHO), provides a unified 
framework for identifying and categorizing diseases through alphanumeric codes, enabling efficient cross-regional and 
cross-institutional data analysis **[1]**.

Despite its advantages, the manual mapping of EMR texts to ICD codes is a resource-intensive and error-prone process. 
Automating this process can significantly enhance accuracy and consistency, providing reliable data for clinical research 
and healthcare management. Recent advancements in deep learning have shown promise in addressing complex classification tasks. 
Among these, Bidirectional Encoder Representations from Transformers (BERT) has demonstrated robust 
performance in various natural language processing applications **[2]**.

In the context of Chinese medical records, fine-tuning pre-trained BERT models offers a promising approach to capturing 
the semantic nuances of clinical texts. This study leverages data from the CCL2025—Chinese Electronic Medical Record ICD 
Diagnostic Coding Evaluation Competition to develop an automated ICD diagnostic coding classification system. The proposed 
method achieves an approximate classification accuracy of 71.25% on the test set, highlighting its potential to 
streamline coding workflows and improve the efficiency of medical data management.