# fraud_detection_kaggle
This project tackles the Kaggle challenge Credit Card Fraud Detection:
https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203

Various machine learning methods are tested for solving this issue:
- K nearest neighbors (knn)
- Gaussian naive Bayes
- Bernoulli naive bayes
- Random forest
- Multilayered perceptron
- xgboost

Xgboost provides the best accuracy on the test set. Error analysis has been conducted for this model.


# How to Run:
- Clone the repository and `cd` into it.
- Install the requirements by running `pip install -r requirements.txt`.
- In your terminal, run `export PYTHONPATH=$PYTHONPATH:$(pwd)` to add the current directory to your `PYTHONPATH`.
