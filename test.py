import pandas as pd
import numpy as np
import keras

df = pd.read_csv('data-training.csv/data-training.csv')

from matplotlib import pyplot as plt

df = df.fillna(0)

#plt.plot(range(1000), df['y'][:1000])
#plt.show()

#convert data set to numpy array
float_data = df.values
X = float_data[:,:-1]
y = float_data[:,-1]

#generator yielding time series samples and their targets
def generator(data, lookback, min_index, max_index, batch_size, shuffle = False):
    if max_index is None:
        max_index = len(data)-1
    i = min_index + lookback
    X = data[:,:-1]
    y = data[:,-1]
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size = batch_size)
        else:
            if i+batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i+batch_size,max_index))
            i += len(rows)

        samples = np.zeros((len(rows),lookback, X.shape[-1]))
        targets =  np.zeros((len(rows),))
        for j,row in enumerate(rows):
            indices = range(rows[j]-lookback, rows[j])
            #print(indices)
            samples[j] = X[indices]
            targets[j] = y[rows[j]-1]
            #print(rows[j])
        yield samples, targets

#preparing the training, validation, and test generators
lookback = 10
batch_size = 128

train_gen = generator(float_data,
                      lookback = lookback,
                      min_index = 0,
                      max_index = 2500000,
                      shuffle = True,
                      batch_size = batch_size)

val_gen = generator(float_data,
                      lookback = lookback,
                      min_index = 2500001,
                      max_index = None,
                      batch_size = batch_size)

# test_gen = generator(float_data,
#                       lookback = lookback,
#                       min_index = 2500001,
#                       max_index = None,
#                       batch_size = batch_size)

val_steps = (len(float_data)-2500001-lookback)//batch_size
#test_steps = (len(float_data)-2500001-lookback)//batch_size

#training and evaluating a GRU-based model
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(64,
                     input_shape = (None, float_data.shape[-1]-1)))
model.add(layers.Dense(1))

model.compile(optimizer = RMSprop(lr = 0.0001), loss = 'mean_squared_error')

history_GRU = model.fit_generator(train_gen,
                              steps_per_epoch = 500,
                              epochs = 20,
                              validation_data = val_gen,
                              validation_steps = val_steps)

model.save('XTXStarterKit-master/python/GRU_model.h5')

#evaluate the train data
#be careful take very long
mse = 0
for i in range(10000):
    pred = model.predict(float_data[i:i+10,:-1][np.newaxis])[0][0]
    mse += 1/10000*(pred-y[i+9])**2
print(mse)

#using bidirectional RNN
model = Sequential()
model.add(layers.Bidirectional(layers.LSTM(64,
                                           activation = 'relu'),
                                           input_shape = (None, float_data.shape[-1]-1)))
model.add(layers.Dense(1))

model.compile(optimizer = RMSprop(), loss = 'mean_squared_error')

history_bi = model.fit_generator(train_gen,
                                 steps_per_epoch = 500,
                                 epochs = 20,
                                 validation_data = val_gen,
                                 validation_steps = val_steps)

#evaluate the whole model
data_gen = generator(float_data,
                      lookback = lookback,
                      min_index = 0,
                      max_index = None,
                      batch_size = batch_size)
steps = (len(float_data)-lookback)//batch_size
scores = model.evaluate_generator(data_gen, steps = steps)



#plot the validation loss
import matplotlib.pyplot as plt

loss = history_GRU.history['loss']
val_loss = history_GRU.history['val_loss']

epochs = range(1,len(loss)+1)
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#compare performance with lightgbm
import lightgbm as lgb
X_train = float_data[:2000000,:-1]
y_train = float_data[:2000000,-1]
X_test = float_data[2000001:2500000,:-1]
y_test = float_data[2000001:2500000,-1]

d_train = lgb.Dataset(X_train, label = y_train)

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'goss'
params['objective'] = 'regression'
params['metric'] = 'mse'
params['sub_feature']= 0.8
params['num_leaves']= 20
params['min_data']= 20
params['max_depth'] = 10

clf = lgb.train(params, d_train, 500)

#output train and test metric
from sklearn.metrics import mean_squared_error
pred_train = clf.predict(X_train)
print('The train mse is', mean_squared_error(y_train,pred_train))
pred_test = clf.predict(X_test)
print('The test mse is', mean_squared_error(y_test,pred_test))
------------------------------------------------------------------------


