from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from xgboost import XGBClassifier
from training_prepare import data_prepare
from preprocessing import data_processing
import joblib



def train(X_train, y_train, model_name):
    """trains a model and saves it

    Args:
        X_train (pandas dataframe): data
        y_train (pandas dataframe): label
        model_name (str): model name

    Returns:
        sklearn model: trained model
    """

    if model_name == 'dummy':
        model = DummyClassifier(strategy="most_frequent")
    elif model_name == 'random_forest':
        model = RandomForestClassifier()
    elif model_name == 'bernoulli_naive_bayes':
        model = BernoulliNB()
    elif model_name == 'gaussian_naive_bayes':
        model = GaussianNB()
    elif model_name == 'xgboost':
        model = XGBClassifier()
    else:
        raise ValueError('model name not found')

    model.fit(X_train, y_train)
    print("saving model...")
    file_name = ''.join([model_name, '_user'])
    model_file = open(f'models/{file_name}.pkl', 'wb')
    joblib.dump(model, model_file)
    model_file.close()


if __name__ == '__main__':

    print('loading data...')
    transaction = data_processing('data/train_transaction.csv')
    identity = data_processing('data/train_identity.csv')
    print('data loaded...')

    X_train, X_test, y_train, y_test = data_prepare(transaction, identity)
    print('training data created...')

    train(X_train, y_train, 'xgboost')
    print('model trained...')


