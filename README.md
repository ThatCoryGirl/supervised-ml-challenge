# supervised-ml-challenge
Python, Pandas, supervised machine learning, 

# **Background**

In this Challenge, you’ll use various techniques to train and evaluate a model based on loan risk. You’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

# **Instructions**

The instructions for this Challenge are divided into the following subsections:

   - Split the Data into Training and Testing Sets

   - Create a Logistic Regression Model with the Original Data

   - Write a Credit Risk Analysis Report

**Split the Data into Training and Testing Sets**

Open the starter code notebook and use it to complete the following steps:

   - Read the `lending_data.csv` data from the Resources folder into a Pandas DataFrame.

   - Create the labels set (`y`) from the “loan_status” column, and then create the features (`X`) DataFrame from the remaining columns.

   - **NOTE** A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.

   - Split the data into training and testing datasets by using `train_test_split`.

**Create a Logistic Regression Model with the Original Data**

Use your knowledge of logistic regression to complete the following steps:

   - Fit a logistic regression model by using the training data (`X_train` and `y_train`).

   - Save the predictions for the testing data labels by using the testing feature data (`X_test`) and the fitted model.

   - Evaluate the model’s performance by doing the following:

      - Generate a confusion matrix.

      - Print the classification report.

   - Answer the following question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

# **Summary and Analysis**

## Overview of the Analysis:

The analysis performed in this machine learning challenge aimed to build a predictive model to assess credit risk based on financial information.

- Purpose of the Analysis:

   - The purpose was to develop a machine learning model capable of predicting credit risk based on various financial attributes. This model could assist financial institutions in evaluating loan applications and assessing the likelihood of loan defaults.

- Financial Information:

   - The dataset contained various financial attributes such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, total debt, and loan status. The task was to predict the loan status, which likely represented whether a loan is classified as healthy or high-risk.

- Variables to Predict:

   - The main variable to predict was the loan status, which had binary values representing healthy (0) or high-risk (1) loans. These variables were predicted based on other financial attributes provided in the dataset.

- Stages of the Machine Learning Process:

   - Data Preparation: Loading the dataset, examining its structure, handling missing values or outliers, and preparing features and labels for modeling.

   - Model Training: Splitting the data into training and testing sets, choosing a suitable algorithm (ie: Logistic Regression), and training the model on the training data.

   - Model Evaluation: Evaluating the trained model's performance using metrics such as confusion matrix, precision, recall, F1-score, and accuracy.

   - Model Interpretation: Analyzing the model's performance, understanding which features are important for prediction, and identifying areas for improvement.

- Methods Used:

   - Logistic Regression: This is a widely used classification algorithm for binary classification tasks like this one. Logistic Regression models the probability that a given input belongs to a particular class.

   - Train-Test Split: This is used to split the dataset into training and testing sets, allowing for model evaluation on unseen data.

   - Confusion Matrix: This is used to evaluate the performance of the classification model by comparing predicted labels with actual labels.

   - Classification Report: This provides a summary of important classification metrics (precision, recall, F1-score) for each class, as well as overall accuracy.

Overall, the analysis involved preparing the data, training and evaluating a Logistic Regression model, and interpreting the results to assess its effectiveness in predicting credit risk based on financial information.

## Results:

Machine Learning Model 1 Results:

   - Accuracy Score: 0.95
     
   - Precision Score:
     
      - Class 0 (Healthy Loan): 0.96
        
      - Class 1 (High-Risk Loan): 0.85
        
   - Recall Score:
     
      - Class 0 (Healthy Loan): 0.97
        
      - Class 1 (High-Risk Loan): 0.76

The supervised machine model achieved an accuracy score of 0.95, indicating that it correctly predicted 95% of the labels in the testing dataset. The precision score for Class 0 (Healthy Loan) is 0.96, indicating that 96% of the loans predicted as healthy were actually healthy. For Class 1 (High-Risk Loan), the precision score is 0.85, indicating that 85% of the loans predicted as high-risk were actually high-risk. The recall score for Class 0 is 0.97, indicating that 97% of the actual healthy loans were correctly identified. For Class 1, the recall score is 0.76, indicating that 76% of the actual high-risk loans were correctly identified.

Machine Learning Model 2 Results:

   - Accuracy Score: 0.92

   - Precision Score:

      - Class 0 (Healthy Loan): 0.94

      - Class 1 (High-Risk Loan): 0.81

   - Recall Score:

      - Class 0 (Healthy Loan): 0.89

      - Class 1 (High-Risk Loan): 0.73

Model 2 achieved an accuracy score of 0.92, indicating that it correctly predicted 92% of the labels in the testing dataset. The precision score for Class 0 (Healthy Loan) is 0.94, indicating that 94% of the loans predicted as healthy were actually healthy. For Class 1 (High-Risk Loan), the precision score is 0.81, indicating that 81% of the loans predicted as high-risk were actually high-risk. The recall score for Class 0 is 0.89, indicating that 89% of the actual healthy loans were correctly identified. For Class 1, the recall score is 0.73, indicating that 73% of the actual high-risk loans were correctly identified.

Based on these results, Model 2 performs slightly worse compared to Model 1 in terms of accuracy, precision, and recall. However, it still provides valuable predictions, and its performance may be acceptable depending on the specific requirements.


## In Conclusion:

Based on these results, it seems that Machine Learning Model 1 performs better overall compared to Model 2. Model 1 has a higher accuracy score, higher precision scores for both classes, and higher recall scores for Class 0 (healthy loans) and Class 1 (high-risk loans).

However, the performance of the models may depend on the specific problem we are trying to solve. In the context of predicting credit risk, it is crucial to consider the consequences of false positives and false negatives. For instance:

   - Predicting 0's (Healthy Loans): If false positives (predicting a loan as healthy when it's high-risk) are more costly, then Model 1 might be preferred due to its higher precision for Class 0.

   - Predicting 1's (High-Risk Loans): If false negatives (predicting a high-risk loan as healthy) are more costly, then Model 2 might be preferred due to its higher recall for Class 1.

Therefore, the choice of model depends on the specific requirements and priorities of the problem at hand. If precision is more important, Model 1 may be preferred. If recall is more important, Model 2 may be preferred. Alternatively, a trade-off between precision and recall can be achieved by adjusting the model's threshold or by using a different evaluation metric, such as F1-score, which balances precision and recall.

# **Citations**

Data for this dataset was generated by edX Boot Camps LLC, and is intended for educational purposes only.

Instructor: [Othmane Benyoucef](https://www.linkedin.com/in/othmane-benyoucef-219a8637/)
