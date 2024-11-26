# Online Payment Fraud Detection Model : A Model Built To Determine The Integrity of Online Transactions.

## Introduction
The aim of this project was to build a model that could classify payments made online as either fraudulent or normal, making use of the machine learning module Scikit-Learn. The whole process involved importing libraries, data cleaning and transformation, data exploration, analysis and visualization, model fitting and validation and can be observed in this [Jupyter Notebook](). The dataset used was obtained from [kaggle](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection), it consisted of eleven(11) columns and 6362620 rows:

**Step**: The time of the transaction in hours.

**Type**: Method used to make payment for the transaction.

**Amount**: The amount paid.

**NameOrig**: The name of the sender.

**OldbalanceOrg**: The initial account balance of the sender before the transaction.

**NewbalanceOrig**: The final account balance of the sender after the transaction.

**NameDest**: The name of the recipient.

**OldbalanceDest**: The initial account balance of the recipient before the transaction.	

**NewbalanceDest**: The final account balance of the recipient after the transaction.	

**IsFraud**: Binary classification for transactions, 0 for normal transaction, 1 for fraud.

**IsFlaggedFraud**: Transactions flagged by the recipient to be fraud.

## Importing Libraries
I started out by importing libraries needed for this project such as Pandas, NumPy, Seaborn, Pyplot from MatPlotLib and several libraries from the Scikit-Learn module.

## Data Evaluation and Cleaning
I proceeded to inspect my dataset after loading it into a DataFrame, checking the number of rows and columns(shape), the datatypes and null-count for each column in the data. I also dropped some columns that were not needed for building the model.

## Data Analysis and Visualization
I performed an exploratory analysis to uncover trends within the data which are stated below;
- The most commonly used payment method  was 'Cash_out'
- Most fradulent transactions were present in this same 'Cash_out' method. 
- There was a higher occurence of normal transactions compared to fraudulent ones making the dataset very imbalanced.
- After creating a class to categorize how expensive each transaction was, I discovered that only the least expensive transactions resulted in fraud when compared to 
  the highly and ultra expensive classes of transactions. So only less expensive transactions were fraudulent.

I created a column from the 'old_bal_org' column and 'new_bal_orig' called 'origin_balance_difference (OBD)', I compared this with the amount column and I observed a consistency in their values. I did the same for the 'old_bal_dest' and 'new_bal_dest' creating a 'destination_balance_difference (DBD)' column, but the values derived as the difference between both columns were not consistent with the amount column, giving the impression that the new balance destination column was not correct for all entries in the dataset.

I also decided to check the correlation between the 'is_fraud' column and every other column in the data, apart from 'is_fraud' itself, the column with the most strong correlation was 'amount'. I categorized the transaction_types into figures to enable the inclusion of this column in the model training and fitting. Then I proceeded to drop the 'OBD', 'DBD' and 'amount_class' columns as they were not needed for the remaining stages of the project.

## Model Fitting and Validation
I started out this process by scaling all feature columns except 'type' using Standard Scaler, then I concatenated the 'type' column with these scaled columns. I split my dataset into 80% train_set and 20% test_set, after also specifying my label column which was the 'is_fraud'column.
I instantiated and fit the Linear Regression, Decision Tree Classifier and XGB Classifier models with the train_data, made predicions on the test_data with these models, then I checked the performance metrics of the models. Due to the imbalance of the data, I decided to consider other techniques to improve these models.
### Undersampling
Based on the performance of the models, I decided to use this method to improve the models performance by taking a random sample of the same size as that of fraudulent transactions from the non-fraudulent data and I concatenated this with the fraudulent data. After scaling, I trained the models using this new data and I noticed a very significant improvement in their performance.
### Oversampling 
I applied Synthetic Minority Over-sampling Technique (SMOTE) to generate samples in the minority(fraud) class so as to balance the dataset here, then I instantiated, fit the models and made predicions on the test_data with them. The XGB Classifier and the Decision Tree Classifier were of best performance compared to the Logistic Regression model, with closely valued metrics. decided to go ahead with the XGB Classifier model with an accuracy value of 0.992696287279367, a precision value of 0.9879007864488808, a recall value of 0.9975565058032987 and an F1 Score of 0.9927051671732523.

## Conclusion
I imported Joblib to save and load this oversampled model. After which I tested out the model by passing a record of feature values to predict if it will be fraudulent or not, and it worked quite well. I also saved the Standard Scaler instance used in scaling the features to a file. 
