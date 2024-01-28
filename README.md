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

Xgboost provides the best accuracy on the test set. Error analysis has been conducted for this model.


# How to Run:
- Clone the repository and cd into it.
- Download the train_transacation.csv and test_transaction.csv files from https://www.kaggle.com/competitions/ieee-fraud-detection/data and add them to the data folder.
- Install the requirements by running pip install -r requirements.txt.
- In your terminal, run export PYTHONPATH=$PYTHONPATH:$(pwd) to add the current directory to your PYTHONPATH.
- run python src/run.py --run_type <run type mode> --model_name <model_name>
<run type mode> : 'train': trains the specified model on data / 'eval': evaluates the saved model on test data (default) / 'compare': evaluates all saved models on test data and compares their accuracy and f1 score
<model name> : 'dummy' / 'random_forest' / 'bernoulli_naive_bayes' / 'gaussian_naive_bayes' / 'xgboost' (default)
