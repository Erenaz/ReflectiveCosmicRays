import os
import pickle
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

import matplotlib
matplotlib.use('Agg')

round = '4thpass'
path = f'Code/data/{round}/'
station = 13  # Change this value to match the station you are working with

# Change this value to control how many times the simulation file is used
simulation_multiplier = 2  # Use the simulation file twice for training

# Get a list of the RCR files
RCR_files = glob(os.path.join(path, "ReflCR_5996events_part0.npy"))
RCR = np.empty((0, 4, 256))
for file in RCR_files:
    RCR_data = np.load(file)[0:5000, 0:4]
    RCR_data = np.vstack([RCR_data] * simulation_multiplier)  # Stack the data multiple times
    RCR = np.concatenate((RCR, RCR_data))

TrainCut = len(RCR)

# Get a list of all the Noise files
Noise_files = glob(os.path.join(path, f"Station{station}_Data_*_part*.npy"))
Noise = np.empty((0, 4, 256))
for file in Noise_files:
    Noise = np.concatenate((Noise, np.load(file)[0:50000, 0:4]))

index = np.arange(0, len(Noise), 1)
np.random.shuffle(index)
Noise = Noise[index]
Noise = Noise[0:TrainCut, 0:4]

#make signal the same shape as the noise data, if needed
#Reuse one set multiple times to match larger dataset of the other
# signal = np.vstack((signal,signal,signal,signal))
# signal = signal[0:noise.shape[0]]

print(RCR.shape)
#print(Nu.shape)
print(Noise.shape)

x = np.vstack((RCR, Noise))  # shape is (200000, 1, 240)
  
n_samples = x.shape[2]
n_channels = x.shape[1]
x = np.expand_dims(x, axis=-1)
#Zeros are noise, 1 signal
#y is output array
y = np.vstack((np.zeros((RCR.shape[0], 1)), np.ones((Noise.shape[0], 1))))
s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]
print(x.shape)

BATCH_SIZE = 32
#Iterate over many epochs to see which has lowest loss
#Then change epochs to be at lowest for final result
EPOCHS = 50

#This automatically saves when loss increases over a number of patience cycles
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
def training(j):
    model = Sequential()
    model.add(Conv2D(10, (4, 10), activation='relu', input_shape=(n_channels, n_samples, 1)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x, y, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks_list)

    # Save the history as a pickle file
    with open(f'Code/h5_models/13/history_{j}_{simulation_multiplier}.pickle', 'wb') as f:
        pickle.dump(history.history, f)

    # Plot the training and validation loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Model {j+1}: Simulation File Used {simulation_multiplier} Times')

    # Save the loss plot as an image file
    plt.savefig(f'Code/h5_models/13/loss_plot_{j}_{simulation_multiplier}.png')

    # Plot the training and validation accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Model {j+1}: Simulation File Used {simulation_multiplier} Times')

    # Save the accuracy plot as an image file
    plt.savefig(f'Code/h5_models/13/accuracy_plot_{j}_{simulation_multiplier}.png')

    plt.show()

    model.summary()

    # Evaluate the model on the validation set
    val_loss, val_acc = model.evaluate(x[-int(0.2 * len(x)):], y[-int(0.2 * len(y)):], verbose=0)
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_acc}')

    #input the path and file you'd like to save the model as (in h5 format)
    model.save(f'Code/h5_models/13/{round}_trained_CNN_1l-10-8-10_do0.5_fltn_sigm_valloss_p4_measNoise0-20k_0-5ksigNU-Scaled_shuff_monitortraining_{j}_{simulation_multiplier}.h5')
  
#can increase the loop for more trainings is you want to see variation
for j in range(1):
  training(j)
