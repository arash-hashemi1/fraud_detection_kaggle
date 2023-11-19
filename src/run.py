
from preprocessing import data_processing
from training_prepare import data_prepare
from model_train import train
from model_eval import eval
from model_compare import compare
import argparse


parser = argparse.ArgumentParser(description='train, eval or compare models along with model name')
parser.add_argument('--run_type', type=str, default='eval', help='train, eval or compare')
parser.add_argument('--model_name', type=str, default='xgboost', help='model name')
args = parser.parse_args()

run_type = args.run_type
model_name = args.model_name

print('loading data...')


transaction = data_processing('data/train_transaction.csv')
identity = data_processing('data/train_identity.csv')
print('data loaded...')

X_train, X_test, y_train, y_test = data_prepare(transaction, identity)
print('training data created...')

if run_type == 'train':
    train(model_name, X_train, y_train)
    print('model trained...')

elif run_type == 'eval':
    score, f1 = eval(model_name, X_test, y_test)
    print(f'score: {score: .4f}, f1: {f1: .4f}')

elif run_type == 'compare':
    compare()
    print('model comparison finished...')

else:
    raise ValueError('run type not found!')

print('run finished...')



