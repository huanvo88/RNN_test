import pandas as pd
import numpy as np
import lightgbm as lgb

df = pd.read_csv('data-training.csv/data-training.csv')

#fill in missing values with zeros
df = df.fillna(0)
X = df.drop(['y'], axis = 1)
y = df['y']

#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)

#train a lightgbm model
d_train = lgb.Dataset(X, label = y)

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'goss'
params['objective'] = 'regression'
params['metric'] = 'mse'
params['sub_feature']= 0.8
params['num_leaves']= 20
params['min_data']= 20
params['max_depth'] = 10

clf = lgb.train(params, d_train, 100)

#output train and test metric
from sklearn.metrics import mean_squared_error
pred_train = clf.predict(X_train)
print('The train mse is', mean_squared_error(y_train,pred_train))
pred_test = clf.predict(X_test)
print('The test mse is', mean_squared_error(y_test,pred_test))

import pickle
path = 'XTXStarterKit-master/python/model.pickle'
with open(path, 'wb') as file:
    pickle.dump(clf,file)

model = pd.read_pickle(path)

