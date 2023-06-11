import os
import numpy as np
from numpy import save, load
import keras
import time
#can do tensorflow.keras if they have tensorflow
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import random
import datetime
import pandas as pd
from glob import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def plotTrace(traces, title, saveLoc, sampling_rate=2, show=False):
    #Sampling rate should be in GHz

    x = np.linspace(1, int(256 / sampling_rate), num=256)
    x_freq = np.fft.rfftfreq(len(x), d=(1 / sampling_rate*units.GHz)) / units.MHz

    fig, axs = plt.subplots(nrows=4, ncols=2, sharex=False)
    for chID, trace in enumerate(traces):
        ax[chID][0].plot(x, trace)
        ax[chID][1].plot(x_freq, np.abs(fft.time2freq(trace, sampling_rate*units.GHz)))

    axs[3][0].set_xlabel('time [ns]',fontsize=18)
    axs[3][1].set_xlabel('Frequency [MHz]',fontsize=18)

    for chID, trace in enumerate(traces):
        axs[chID][0].set_ylabel(f'ch{chID}',labelpad=10,rotation=0,fontsize=13)
        # axs[i].set_ylim(-250,250)
        axs[chID][0].set_xlim(-3,260 / sampling_rate)
        axs[chID][1].set_xlim(-3, 500)
        axs[chID][0].tick_params(labelsize=13)
        axs[chID][1].tick_params(labelsize=13)
    axs[0][0].tick_params(labelsize=13)
    axs[0][1].tick_params(labelsize=13)
    axs[0][0].set_ylabel(f'ch{0}',labelpad=3,rotation=0,fontsize=13)
    axs[c][0].set_xlim(-3,260 / sampling_rate)
    axs[c][1].set_xlim(-3, 500)

    fig.text(0.03, 0.5, 'voltage [V]', ha='center', va='center', rotation='vertical',fontsize=18)
    plt.xticks(size=13)
    plt.suptitle(title)

    if show:
        plt.show()
    else:
        plt.savefig(saveLoc, format='png')
    return

def plotTimeStrip(times, vals, title, saveLoc):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    plt.gca().xaxis.set_major_locator(locator)
    plt.scatter(times,vals, alpha=0.5)
    plt.gcf().autofmt_xdate()
#Lower limit set to install date of earliest station to remove error/bad times
#Need to remove to see the effect of bad datetimes
    plt.xlim(datetime(2013, 7, 1), plt.xlim()[1])   
    plt.title(title)
    plt.savefig(saveLoc, format='png')
    
    return

def data_generator(files, chunk_size=50000):
    for file in files:
        # Load the noise data from the current file
        data_file = np.load(file)
        num_chunks = data_file.shape[0] // chunk_size
        if data_file.shape[0] % chunk_size:
            num_chunks += 1

        for chunk in range(num_chunks):
            start = chunk * chunk_size
            end = min(start + chunk_size, data_file.shape[0])

            # Yield the chunk from the data
            yield data_file[start:end]

# input path and file names and station ID.
round = '4thpass'
path = f'Code/data/{round}/'
station = 14  # Change this value to match the station you are working with

# Change this value to control how many times the simulation file is used
simulation_multiplier = 1  

# Get a list of all the Noise files
Noise_files = glob(os.path.join(path, f"Station{station}_Data_*_part*.npy"))
# Get a list of all the DateTime files
DateTime_files = glob(os.path.join(path, f"DateTime_Station{station}_Data_*_part*.npy"))

# Create generators
noise_generator = data_generator(Noise_files)
datetime_generator = data_generator(DateTime_files)

# Load the model
model = keras.models.load_model(f'Code/h5_models/New/14/{round}_CNN_2l-10-4-10-5-1-10_do0.5_fltn_sigm_valloss_p4_measNoise0-20k_0-5ksigNU-Scaled_shuff_monitortraining_0_{simulation_multiplier}.h5')

# Collectors for plot values
all_max_amps = []
all_prob_Noise = []
all_datetimes = []

# Predict in chunks and process the data immediately to free memory
for datetime_chunk, noise_chunk in zip(datetime_generator, noise_generator):
    noise_chunk = np.reshape(noise_chunk, (noise_chunk.shape[0], noise_chunk.shape[1], noise_chunk.shape[2], 1))
    prob_Noise = model.predict(noise_chunk)
    datetime_chunk = np.vectorize(datetime.datetime.fromtimestamp)(datetime_chunk)

    # Process the prediction immediately
    plotRandom = True
    if plotRandom == True:
        for iE, Noi in enumerate(noise_chunk):
            output = 1 - prob_Noise[iE][0]
            if output > 0.95:
                print(f'output {output}')
                plotTrace(Noi, f"Noise {iE}, Output {output:.2f}, " + datetime_chunk[iE].strftime("%m-%d-%Y, %H:%M:%S"),f"Code/data/4thpass/Station_14/Noise/Noise_{iE}_Output_{output:.2f}_Station{station}.png")

    max_amps = np.zeros(len(prob_Noise))
    for iC, trace in enumerate(noise_chunk):
        max_amps[iC] = np.max(trace)

    # Add max_amps and prob_Noise to collectors
    all_prob_Noise.extend(prob_Noise.flatten())
    all_max_amps.extend(max_amps.tolist())
    all_datetimes.extend(datetime_chunk.tolist())

    # Clear memory
    del noise_chunk

# Load and predict the RCR data
RCR_files = glob(os.path.join(path, "ReflCR_5730events_part0.npy"))
RCR = np.empty((0, 4, 256))
for file in RCR_files:
    RCR_data = np.load(file)[5000:, 0:4]
    RCR_data = np.vstack([RCR_data] * simulation_multiplier)  # Stack the data multiple times
    RCR = np.concatenate((RCR, RCR_data))

RCR = np.reshape(RCR, (RCR.shape[0], RCR.shape[1], RCR.shape[2], 1))
prob_RCR = model.predict(RCR)

# Process the RCR prediction
plotRandom = True
if plotRandom == True:
    for iE, rcr in enumerate(RCR):
        if iE % 1000 == 0:
            output = 1 - prob_RCR[iE][0]
            if output > 0.95:
                plotTrace(rcr, f"RCR {iE}, Output {output:.2f}",f"Code/data/4thpass/Station_14/RCR/RCR_{iE}_Output_{output:.2f}_Station{station}.png")

max_amps_RCR = np.zeros(len(prob_RCR))
for iC, trace in enumerate(RCR):
    max_amps_RCR[iC] = np.max(trace)

prob_RCR = 1 - prob_RCR

plt.scatter(all_max_amps, all_prob_Noise, color='blue', label='Noise')
plt.scatter(max_amps_RCR, prob_RCR, color='orange', label='SimRCR')
plt.ylabel('Network Output - 1 = RCR')
plt.xlabel('Max amp of channels')
plt.legend()
plt.title(f'Station 14 Training')
plt.grid(True)
plt.savefig(path+f'Station_14/MaxAmpsOutputStation{station}_M{simulation_multiplier}.png', format='png')
plt.clf()

haveTimes = True
if haveTimes:
    plotTimeStrip(datetime_chunk, prob_Noise, f'Station 14', saveLoc=path+'plots/Stn14_TimeStrip.png')

fig = plt.figure()
ax = fig.add_subplot(111)

dense_val = False
ax.hist(prob_Noise, bins=20, range=(0, 1), histtype='step', color='red', linestyle='solid', label='Station_14Data', density=dense_val)
ax.hist(prob_RCR, bins=20, range=(0, 1), histtype='step',color='blue', linestyle='solid',label='SimRCR',density = dense_val)

plt.xlabel('network output', fontsize=18)
plt.ylabel('events', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.yscale('log')
plt.title('Station 14')
handles, labels = ax.get_legend_handles_labels()
new_handles = [Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle()) for h in handles]
plt.legend(loc='upper center', handles=new_handles, labels=labels, fontsize=18)
plt.savefig(path+f'Station_14/Station{station}AnalysisOutput_M{simulation_multiplier}.png', format='png')
