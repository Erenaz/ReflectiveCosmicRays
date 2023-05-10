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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def plotTrace(traces, title, saveLoc, show=False):
    f, ax = plt.subplots(4,1)
    for chID, trace in enumerate(traces):
        ax[chID].plot(trace)
    plt.suptitle(title)
    if show:
        plt.show()
    else:
        plt.savefig(saveLoc, format='png')
    return

def plotTimeStrip(times, vals, title, saveLoc):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.scatter(times,vals, alpha=0.5)
    plt.gcf().autofmt_xdate()
    plt.title(title)
    plt.savefig(saveLoc, format='png')

#input path and file names and station ID.
round = '4thpass'
path = f'Code/data/{round}/'
station = 13  # Change this value to match the station you are working with

# Change this value to control how many times the simulation file is used
simulation_multiplier = 1  # Use the simulation file twice for training

# Get a list of all the RCR files
RCR_files = glob(os.path.join(path, "ReflCR_*events_part0.npy"))
RCR = np.empty((0, 4, 256))
for file in RCR_files:
    RCR_data = np.load(file)[5000:, 0:4]
    RCR_data = np.vstack([RCR_data] * simulation_multiplier)  # Stack the data multiple times
    RCR = np.concatenate((RCR, RCR_data))
    
# Get a list of all the noise files
Noise_files = glob(os.path.join(path, f"Station{station}_Data_*_part*.npy"))
Noise = np.empty((0, 4, 256))
for file in Noise_files:
    Noise = np.concatenate((Noise, np.load(file)[50000:,0:4]))

RCR = np.reshape(RCR, (RCR.shape[0], RCR.shape[1],RCR.shape[2],1))
Noise = np.reshape(Noise, (Noise.shape[0], Noise.shape[1],Noise.shape[2],1))

haveTimes = False
if haveTimes == True:
    #I don't think this needs reshaping, maybe it does like above
    times = np.load(os.path.join(path, "Put_time_file_here"))[10000:]
    for iT, t in enumerate(times):
        times[iT] = datetime.utcfromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')

#input path to trained h5 model
model = keras.models.load_model(f'Code/h5_models/13/{round}_trained_CNN_1l-10-8-10_do0.5_fltn_sigm_valloss_p4_measNoise0-20k_0-5ksigNU-Scaled_shuff_monitortraining_0_{simulation_multiplier}.h5')

prob_RCR = model.predict(RCR)
prob_Noise = model.predict(Noise)
                
plotRandom = True
if plotRandom == True:
    for iE, Noi in enumerate(Noise):
        output = 1 - prob_Noise[iE][0]
        if output > 0.95:
            print(f'output {output}')
            plotTrace(Noi, f"Noise {iE}, Output {output:.2f}",f"Code/data/4thpass/Station_13/Noise/Noise_{iE}_Output_{output:.2f}_Station{station}.png")
    for iE, rcr in enumerate(RCR):
        if iE % 1000 == 0:
            output = 1 - prob_RCR[iE][0]
            if output > 0.95:
            plotTrace(rcr, f"RCR {iE}, Output {output:.2f} " + times[iE],f"Code/data/4thpass/Station_13/RCR/RCR_{iE}_Output_{output:.2f}_Station{station}.png")
                
max_amps = np.zeros(len(prob_Noise))
for iC, trace in enumerate(Noise):
    max_amps[iC] = np.max(trace)
max_amps_RCR = np.zeros(len(prob_RCR))
for iC, trace in enumerate(RCR):
    max_amps_RCR[iC] = np.max(trace)

prob_Noise = 1 - prob_Noise
prob_RCR = 1 - prob_RCR
plt.scatter(max_amps, prob_Noise, label='StationData')
plt.scatter(max_amps_RCR, prob_RCR, label='SimRCR')
plt.ylabel('Network Output - 1 = RCR')
plt.xlabel('Max amp of channels')
plt.legend()
plt.title(f'Station {station} Training')
plt.savefig(path+f'Station_13/MaxAmpsOutputStation{station}_M{simulation_multiplier}.png', format='png')
plt.clf()
#plt.show()

if haveTimes:
    plotTimeStrip(times, prob_Noise, f'Station 13', saveLoc=path+'plots/Stn13_TimeStrip.png')

fig = plt.figure()
ax = fig.add_subplot(111)

dense_val = False
ax.hist(prob_Noise, bins=20, range=(0, 1), histtype='step', color='red', linestyle='solid', label='Station_13Data', density=dense_val)
ax.hist(prob_RCR, bins=20, range=(0, 1), histtype='step',color='blue', linestyle='solid',label='SimRCR',density = dense_val)


plt.xlabel('network output', fontsize=18)
plt.ylabel('events', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.yscale('log')
plt.title('Station {station}')
handles, labels = ax.get_legend_handles_labels()
new_handles = [Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle()) for h in handles]
plt.legend(loc='upper center', handles=new_handles, labels=labels, fontsize=18)
#plt.show()
plt.savefig(path+f'Station_13/Station{station}AnalysisOutput_M{simulation_multiplier}.png', format='png')
