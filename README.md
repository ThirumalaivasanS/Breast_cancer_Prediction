
# Predicting Breast Cancer in a patient

This project aims to detect breast cancer using a Support Vector Machine (SVM) model. The dataset used is the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, which contains information about various features of breast cell nuclei.





## Prerequisites
To run this project, you need to have Python 3.x installed on your system along with the following libraries:
1. pandas
2. numpy
3. matplotlib
4. seaborn
5. scikit-learn



## Necessary imports
1. import pandas as pd
2. import numpy as np
3. import matplotlib.pyplot as plt
4. import seaborn as sns


#### Pre-Modeling Tasks
1. from sklearn.preprocessing import LabelEncoder
2. from sklearn.model_selection import train_test_split
3. from sklearn.preprocessing import StandardScaler


### Modeling
1. from sklearn.svm import SVC


### Evaluation and comparision of the model
1. from sklearn.metrics import accuracy_score
2. from sklearn.metrics import classification_report
3. from sklearn.metrics import precision_score, recall_score, confusion_matrix
4. from sklearn.metrics import roc_auc_score,auc


## Project Overview

The project consists of the following steps:
1.	Load the dataset and perform exploratory data analysis (EDA).
2.	Split the dataset into the majority and minority classes, randomly select a subset of the majority class with the same number of samples as the minority class and concatenate the minority class with the subset of the majority class to create the undersampled dataset.
3.	Perform feature selection and dimensionality reduction.
4.	Encode categorical data.
5.	Separate the independent and dependent variables.
6.	Split the dataset into training and testing sets.
7.	Perform feature scaling.
8.	Train an SVM model on the training set.
9.	Evaluate the performance of the SVM model on the testing set using various metrics such as accuracy, precision, recall, F1 score, and ROC curve.




## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

Contributions are always welcome!

## Credits
This script was created by Thirumalaivasan S.Feel free to modify and use it for your own purposes. If you have any questions or suggestions, please contact me at thirumalaivasan.subramanian@gmail.com