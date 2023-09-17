## load saved models, create score, f1scores and create the comparison table
import numpy as np
import pandas as pd
import joblib
from training_prepare import data_prepare
from preprocessing import data_processing
from model_eval import eval
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def compare():

    """loads saved models, creates score, f1scores and creates the comparison table
    """

    print('loading data...')
    transaction = data_processing('data/train_transaction.csv')
    identity = data_processing('data/train_identity.csv')
    print('data loaded...')

    X_train, X_test, y_train, y_test = data_prepare(transaction, identity)
    print('training data created...')


    model_list = ['dummy',
                'knn',
                'random_forest',
                'bernoulli_naive_bayes',
                'gaussian_naive_bayes',
                'xgboost',
                ]

    model_score = []
    model_f1 = []
    print('starting the loop...')
    for item in model_list:
        score, f1 = eval(item, X_test, y_test)
        # print(f'{type(model).__name__} score: {score: .4f}, f1: {f1: .4f}')
        model_score.append(score)
        model_f1.append(f1)
    print('loop finished...')
    print('creating the table...')
    model_compare = pd.DataFrame({'model': model_list, 'score': model_score, 'f1': model_f1})
    model_compare = model_compare.sort_values(by='f1', ascending=False)
    model_compare = model_compare.set_index('model')
    print('table created...')
    print(model_compare)
    model_compare.to_csv('model_compare.csv')


if __name__ == '__main__':

    compare()
    print('model comparison finished...')