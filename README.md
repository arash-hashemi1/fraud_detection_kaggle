# Credit Card Transaction Fraud Detection (Kaggle)

This project tackles the Kaggle challenge Credit Card Transaction Fraud Detection [(Link)](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203)


<p align="justify">
Different machine learning approaches are experimented with to address this challenge. A dummy classifier is deliberately incorporated as a baseline to assess the training of the models. The selection of these models is based on their common application in classification problems. Specifically, random forest and XGBOOST are chosen for their ability to handle missing values, resilience to variable input scales, and effectiveness in mitigating overfitting.

- Dummy classifier 
- Gaussian naive Bayes
- Bernoulli naive bayes
- Random forest
- Multilayered perceptron
- XGBOOST

Upon evaluation on test data, the XGBOOST model demonstrates the highest accuracy.

</p>

![image](https://github.com/arash-hashemi1/fraud_detection_kaggle/assets/48169508/bb2567ea-713f-4873-9b3a-dd5b7391f852)


Table of contents
=================

- [Feature Engineering](#feature-extraction)
- [Preprocessing ](#preprocessing)
- [Model Comparison](#model-comparison)
- [How to Run](#how-run)



# Feature Engineering

After reviewing the available data, the following features are selected for training as they appeared to be the most relevant for the task of transaction fraud detection. These features encompass both numerical and categorical values.

| Feature                | Description                                                                                      |
|------------------------|--------------------------------------------------------------------------------------------------|
| TransactionDT          | Timedelta from a given reference datetime (not an actual timestamp)                               |
| TransactionAMT         | Transaction payment amount in USD                                                                |
| ProductCD              | Product code, representing the product for each transaction                                       |
| card1 - card6          | Payment card information, including card type, card category, issue bank, country, etc.            |
| addr                   | Address                                                                                          |
| dist                   | Distance                                                                                         |
| P_ and (R__) emaildomain | Purchaser and recipient email domain                                                           |
| C1-C14                 | Counting features, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked. |
| D1-D15                 | Timedelta features, representing days between previous transactions, etc.                          |
| M1-M9                  | Match features, such as names on card and address, etc.                                           |
| Vxxx                   | Vesta engineered rich features, including ranking, counting, and other entity relations.         |


# Preprocessing 
Data cleaning is conducted on both the training and test datasets, involving the extraction of pertinent features from the data frames. Numerous columns contain missing values, and to address this, a random selection from the available values is employed for replacement, thereby preserving the original column distribution. Furthermore, categorical feature columns are transformed using OneHot encoding.

# Model Comparison

The trained models undergo evaluation using the testing data, and the subsequent results are illustrated below. Amongst the models, the XGBOOST model demonstrates the highest accuracy and f1 score.

| Model                   | Score              | F1                |
|-------------------------|--------------------|-------------------|
| xgboost                 | 0.965              | 0.748             |
| random_forest           | 0.962              | 0.712             |
| bernoulli_naive_bayes   | 0.748              | 0.292             |
| gaussian_naive_bayes    | 0.114              | 0.152             |
| dummy                   | 0.918              | 0.0               |


# How to Run
- Clone the repository and cd into it.
- Download the train_transacation.csv and test_transaction.csv files from https://www.kaggle.com/competitions/ieee-fraud-detection/data and add them to the data folder.
- Install the requirements by running pip install -r requirements.txt.
- In your terminal, run export PYTHONPATH=$PYTHONPATH:$(pwd) to add the current directory to your PYTHONPATH.
- run python src/run.py --run_type <run type mode> --model_name <model_name>
<run type mode> : 'train': trains the specified model on data / 'eval': evaluates the saved model on test data (default) / 'compare': evaluates all saved models on test data and compares their accuracy and f1 score
<model name> : 'dummy' / 'random_forest' / 'bernoulli_naive_bayes' / 'gaussian_naive_bayes' / 'xgboost' (default)
