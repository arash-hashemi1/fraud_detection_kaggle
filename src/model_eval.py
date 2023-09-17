from sklearn.metrics import f1_score
from training_prepare import data_prepare
from preprocessing import data_processing
import joblib

def eval(model_name, X_val, y_val):
    """evaluates trained model

    Args:
        model (sklearn model): model_name
        X_val (pandas dataframe): data
        y_val (pandas dataframe): label

        return (float): score, f1_score
    """
    file = open(f'models/{model_name}.pkl', 'rb')
    model = joblib.load(file)
    y_preds = model.predict(X_val)
    score = model.score(X_val, y_val)
    f1 = f1_score(y_val, y_preds)
    file.close()

    return score, f1


if __name__ == '__main__':

    print('loading data...')
    transaction = data_processing('data/train_transaction.csv')
    identity = data_processing('data/train_identity.csv')
    print('data loaded...')

    X_train, X_test, y_train, y_test = data_prepare(transaction, identity)
    print('training data created...')

    score, f1 = eval('xgboost', X_test, y_test)
    print(f'score: {score: .4f}, f1: {f1: .4f}')

