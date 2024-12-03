# Online Payment Fraud Detection Model : A Model Built To Determine The Integrity of Online Transactions.

## Introduction
The aim of this project was to build a model that could classify payments made online as either fraudulent or normal, making use of the machine learning module Scikit-Learn. The whole process involved importing libraries, data cleaning and transformation, data exploration, analysis and visualization, model fitting and validation and can be observed in this [Jupyter Notebook](https://github.com/OluwanifemiAjayi/Fraud_Detection_Model/blob/main/Online_payments_fraud_detection.ipynb). The dataset used was obtained from [kaggle](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection) and it consisted of eleven(11) columns and 6362620 rows:

**Step**: The time of the transaction in hours, one step being equivalent to an hour.

**Type**: The method used to make payment for the transaction.

**Amount**: The amount paid.

**NameOrig**: The name of the sender.

**OldbalanceOrg**: The initial account balance of the sender before the transaction.

**NewbalanceOrig**: The final account balance of the sender after the transaction.

**NameDest**: The name of the recipient.

**OldbalanceDest**: The initial account balance of the recipient before the transaction.	

**NewbalanceDest**: The final account balance of the recipient after the transaction.	

**IsFraud**: Binary classification for transactions, 0 for normal transactions, 1 for fraudulent tansactions.

**IsFlaggedFraud**: Transactions flagged by the recipient to be fraud.

## Importing Libraries
I imported all libraries needed for this project such as Pandas, NumPy, Seaborn, Pyplot from MatPlotLib and several libraries from the Scikit-Learn module.

## Data Evaluation and Cleaning
I inspected my dataset after loading it into a DataFrame, checking the number of rows and columns(shape), the datatypes and null-count for each column in the data. I also dropped some columns that were not needed for building the model.

## Data Analysis and Visualization
I performed an exploratory analysis to uncover trends within the data which are stated below;
- The most commonly used payment method  was 'Cash_out'
- Most fradulent transactions were present in this same 'Cash_out' method. 
- There was a higher occurence of normal transactions compared to fraudulent ones making the dataset very imbalanced.
- The least expensive transactions resulted in fraud when compared to the highly and ultra expensive classes of transactions. So only less expensive transactions 
  were fraudulent.

I created a column from the 'old_bal_org' column and 'new_bal_orig' called 'origin_balance_difference (OBD)', when compared with the amount column,  consistency in their values is observed. This was not the same for 'old_bal_dest' and 'new_bal_dest', as the column derived as the difference between both columns was not consistent with the amount column.

I performed a correlation analysis between the 'is_fraud' column and every other column in the data, apart from 'is_fraud' itself, the column with the most strong correlation was 'amount'. I categorized the transaction_types into figures to enable the inclusion of this column in the model training and fitting. Then I proceeded to drop the 'OBD', 'DBD' and 'amount_class' columns as they were not needed for the remaining stages of the project.

## Model Fitting and Validation
I scaled all feature columns except 'type' using Standard Scaler, then I concatenated the 'type' column with these scaled columns. I split my dataset into 80% train_set and 20% test_set, after also specifying my label column which was the 'is_fraud'column.
I instantiated and fit the Linear Regression, Decision Tree Classifier and XGB Classifier models with the train_data, made predicions on the test_data with these models, then I checked the performance metrics of the models. Due to the imbalance of the data, I decided to consider other techniques to improve these models.
### Undersampling
Based on the initial performance of the models, I employed undersampling to improve the models performance. After scaling, I trained the models using this new undersampled data and I noticed a very significant improvement in their performance.
### Oversampling 
I applied Synthetic Minority Over-sampling Technique (SMOTE) to generate samples in the minority(fraud) class so as to resolve the issue of data imbalance, then I trained the models with the new data. The XGB Classifier and the Decision Tree Classifier were of best performance compared to the Logistic Regression model, also they both had closely valued metrics. I went ahead with the XGB Classifier model with an accuracy value of 0.992696287279367, a precision value of 0.9879007864488808, a recall value of 0.9975565058032987 and an F1 Score of 0.9927051671732523.

## Conclusion
I imported Joblib to save and load this oversampled model, I then tested out the model by passing a record of feature values to predict the transaction's integrity, and it worked quite well. I also saved the Standard Scaler instance used in scaling the features to a file. 
