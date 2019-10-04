import pandas as pd
import numpy as np
import keras

df = pd.read_csv('data-training.csv')

df = df.fillna(0)

#change the order of the columns
columns = ['askRate0','askSize0','bidRate0','bidSize0']
for i in range(1,15):
    columns += ['askRate'+str(i),'askSize'+str(i),'bidRate'+str(i), 'bidSize'+str(i)]
columns += ['y']

df = df[columns]

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
            samples[j] = X[indices]
            targets[j] = y[rows[j]-1]
        samples = samples[:,:,:,np.newaxis]
        yield samples, targets

#preparing the training, validation, and test generators
lookback = 200
batch_size = 32

train_gen = generator(float_data,
                      lookback = lookback,
                      min_index = 0,
                      max_index = 10000,
                      shuffle = False,
                      batch_size = batch_size)

val_gen = generator(float_data,
                      lookback = lookback,
                      min_index = 10001,
                      max_index = 12000,
                      batch_size = batch_size)

test_gen = generator(float_data,
                      lookback = lookback,
                      min_index = 12001,
                      max_index = 13000,
                      batch_size = batch_size)

train_steps = (10000-lookback)//batch_size
val_steps = (12000-10001-lookback)//batch_size
test_steps = (13000-12001-lookback)//batch_size

#building a deep learning model
from keras.models import Sequential, Model
from keras import layers
from keras.optimizers import RMSprop, Adam, SGD

x = layers.Input(shape = (200,60,1))
y = layers.Conv2D(filters = 16,
                        kernel_size = (1,2),
                        strides = (1,2),
                        padding = 'same',
                        activation = 'relu')(x)
y = layers.Conv2D(filters = 16,
                        kernel_size = (4,1),
                        padding = 'same',
                        activation = 'relu')(y)
y = layers.Conv2D(filters = 16,
                        kernel_size = (4,1),
                        padding = 'same',
                        activation = 'relu')(y)
y = layers.Conv2D(filters = 16,
                        kernel_size = (1,2) ,
                        strides = (1,2),
                        padding = 'same',
                        activation = 'relu')(y)
y = layers.Conv2D(filters = 16,
                        kernel_size = (4,1),
                        padding = 'same',
                        activation = 'relu')(y)
y = layers.Conv2D(filters = 16,
                        kernel_size = (4,1),
                        padding = 'same',
                        activation = 'relu')(y)
y = layers.Conv2D(filters = 16,
                        kernel_size = (1,15),
                        #padding = 'same',
                        activation = 'relu')(y)
y = layers.Conv2D(filters = 16,
                        kernel_size = (4,1),
                        padding = 'same',
                        activation = 'relu')(y)
y = layers.Conv2D(filters = 16,
                        kernel_size = (4,1),
                        padding = 'same',
                        activation = 'relu')(y)

#implement inception layer
tower_1 = layers.Conv2D(filters = 32,
                        kernel_size = (1,1),
                        padding = 'same',
                        activation = 'relu')(y)
tower_1 = layers.Conv2D(filters = 32,
                        kernel_size = (3,1),
                        padding = 'same',
                        activation = 'relu')(tower_1)

tower_2 = layers.Conv2D(filters = 32,
                        kernel_size =  (1,1),
                        padding = 'same',
                        activation = 'relu')(y)
tower_2 = layers.Conv2D(filters = 32,
                        kernel_size = (5,1),
                        padding = 'same',
                        activation = 'relu')(tower_2)

tower_3 = layers.MaxPooling2D(pool_size=(3,1),
                              strides = (1,1),
                              padding = 'same')(y)
tower_3 = layers.Conv2D(filters = 32,
                        kernel_size = (1,1),
                        padding = 'same',
                        activation = 'relu')(tower_3)

y = layers.concatenate([tower_1,tower_2,tower_3], axis = 3)

y = layers.Reshape((200,-1))(y)
y = layers.GRU(32)(y)
y = layers.Dense(1)(y)

model = Model(x,y)

model.compile(optimizer = RMSprop(lr = 0.0001), loss = 'mean_squared_error')

history_GRU = model.fit_generator(train_gen,
                              steps_per_epoch = train_steps,
                              epochs = 40,
                              validation_data = val_gen,
                              validation_steps = val_steps)

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
