import os
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
import random
import math

round = '4thpass'
path = f'Code/data/{round}/'

# Get a list of all the RCR files
RCR_files = glob(os.path.join(path, "ReflCR_*_part*.npy"))
RCR = []

for file in RCR_files:
    with open(file, 'rb') as f:
        if RCR == []:
            RCR = np.load(f)
        else:
            RCR = np.concatenate((RCR, np.load(f)))

num_RCR_events = len(RCR)
print('Number of RCR events:', num_RCR_events)

Noise_files = glob(os.path.join(path, "Station13_Data_*_part*.npy"))
all_events = []

# Load all events into a list
for file in Noise_files:
    with open(file, 'rb') as data:
        events = np.load(data)
        all_events.extend(events)

# Randomly select the same number of events as RCR from the list
selected_events = random.sample(all_events, num_RCR_events)

# Convert the list of selected events back to a NumPy array
Noise = np.array(selected_events)

print('RCRShape=', RCR.shape)
#print(Nu.shape)
print('NoiseShape=', Noise.shape)

x_train = np.vstack((RCR, Noise))  # shape is (200000, 1, 240)

n_samples = x_train.shape[2]
n_channels = x_train.shape[1]
x_train = np.expand_dims(x_train, axis=-1)
#Zeros are noise, 1 signal
#y is output array
y_train = np.vstack((np.zeros((RCR.shape[0], 1)), np.ones((Noise.shape[0], 1))))
s = np.arange(x_train.shape[0])
np.random.shuffle(s)
x_train = x_train[s]
y_train = y_train[s]
print('XShape=', x_train.shape)

# Split data into training, validation, and test sets
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

train_size = int(train_ratio * len(x_train))
val_size = int(val_ratio * len(x_train))
test_size = len(x_train) - train_size - val_size

x_val = x_train[train_size:train_size+val_size]
y_val = y_train[train_size:train_size+val_size]

x_test = x_train[train_size+val_size:]
y_test = y_train[train_size+val_size:]

x_train = x_train[:train_size]
y_train = y_train[:train_size]

BATCH_SIZE = 32
#Iterate over many epochs to see which has lowest loss. Then change epochs to be at lowest for final result.
EPOCHS = 100

#This automatically saves when loss increases over a number of patience cycles
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

def training(j):
  model = Sequential()
  #Convolutions have worked better for our data so far than connected NN
  #First num = number of kernel/filters
  #Second pair of numbers is kernel/filter size
  #4 channels, 10 samples wide
  #can widen the samples to get different set of features, could go up to 50 samples
  model.add(Conv2D(10, (4, 10), activation='relu', input_shape=(n_channels, n_samples, 1)))
  model.add(Dropout(0.5))
  model.add(Flatten())
  #If doing 3 options, change activation to 'softmax'
  model.add(Dense(1, activation='sigmoid'))
  model.compile(optimizer='Adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks_list)

  # Evaluate the model on the test set
  test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', test_loss)
  print('Test accuracy:', test_acc)

  model.summary()

  #input the path and file you'd like to save the model as (in h5 format)
  model.save(f'Code/h5_models/{round}_trained_CNN_1l-10-8-10_do0.5_fltn_sigm_valloss_p4_measNoise0-20k_0-5ksigNU-Scaled_shuff_monitortraining_{j}_13.h5')
  
#can increase the loop for more trainings is you want to see variation
for j in range(1):
  training(j)
