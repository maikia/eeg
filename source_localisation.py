import os
import pandas as pd
import pickle
import scipy.sparse as sparse
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import RidgeClassifierCV

# Load the parcel names and the parcel indices
infile = open(os.path.join('data', 'labels.pickle'), 'rb')
labels = pickle.load(infile)
infile.close()

# Load the dataset using Pandas
X_train = pd.read_csv(
    os.path.join('data', 'train.csv')
)
y_train = sparse.load_npz(os.path.join('data', 'train_target.npz'))

ridge = RidgeClassifierCV(class_weight="balanced")
multi_target_ridge = MultiOutputClassifier(ridge, n_jobs=-1)
multi_target_ridge.fit(X_train, y_train.toarray())

# Load test data
X_test = pd.read_csv(
    os.path.join('data', 'test.csv')
)
y_test = sparse.load_npz(os.path.join('data', 'test_target.npz'))
print(multi_target_ridge.score(X_test, y_test.toarray()))
