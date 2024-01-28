# Fraud Detection (Kaggle)
This project tackles the Kaggle challenge Credit Card Fraud Detection:
https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203

Various machine learning methods are tested for solving this issue:
- Dummy classifier 
- Gaussian naive Bayes
- Bernoulli naive bayes
- Random forest
- Multilayered perceptron
- XGBOOST

Upon evaluation on test data, the XGBOOST model demonstrates the highest accuracy.



# Preprocessing 
Data cleaning is performed on both the training and test datasets. Relevant features are extracted from the data frames, and missing values are substituted with a random choice from the available values. Additionally, categorical feature columns undergo a OneHot encoding transformation.

# Model Comparison

The trained models undergo evaluation using the testing data, and the subsequent results are illustrated below. Amongst the models, the XGBOOST model demonstrates the highest accuracy and f1 score.

| Model                   | Score              | F1                |
|-------------------------|--------------------|-------------------|
| xgboost                 | 0.965              | 0.748             |
| random_forest           | 0.962              | 0.712             |
| bernoulli_naive_bayes   | 0.748              | 0.292             |
| gaussian_naive_bayes    | 0.114              | 0.152             |
| dummy                   | 0.918              | 0.0               |


# How to Run:
- Clone the repository and cd into it.
- Download the train_transacation.csv and test_transaction.csv files from https://www.kaggle.com/competitions/ieee-fraud-detection/data and add them to the data folder.
- Install the requirements by running pip install -r requirements.txt.
- In your terminal, run export PYTHONPATH=$PYTHONPATH:$(pwd) to add the current directory to your PYTHONPATH.
- run python src/run.py --run_type <run type mode> --model_name <model_name>
<run type mode> : 'train': trains the specified model on data / 'eval': evaluates the saved model on test data (default) / 'compare': evaluates all saved models on test data and compares their accuracy and f1 score
<model name> : 'dummy' / 'random_forest' / 'bernoulli_naive_bayes' / 'gaussian_naive_bayes' / 'xgboost' (default)
