import datetime
from NuRadioReco.utilities import units
from NuRadioReco.modules import channelResampler as CchannelResampler
from NuRadioReco.modules.ARIANNA import hardwareResponseIncorporator as ChardwareResponseIncorporator
from NuRadioReco.modules import channelTimeWindow as cTWindow
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelStopFilter
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.triggerTimeAdjuster
import NuRadioReco.modules.channelLengthAdjuster
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.modules.io import NuRadioRecoio
import numpy as np
import os
from NuRadioReco.detector import generic_detector
from NuRadioReco.detector import detector
import datetime
import json

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

channelResampler = CchannelResampler.channelResampler()
channelResampler.begin(debug=False)
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
hardwareResponseIncorporator = ChardwareResponseIncorporator.hardwareResponseIncorporator()
hardwareResponseIncorporator.begin(debug=False)
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
correclationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
correclationDirectionFitter.begin(debug=False)
channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()
cTW = cTWindow.channelTimeWindow()
cTW.begin(debug=False)
triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
triggerTimeAdjuster.begin()
channelLengthAdjuster = NuRadioReco.modules.channelLengthAdjuster.channelLengthAdjuster()
channelLengthAdjuster.begin()
# det = detector_sys_uncertainties.DetectorSysUncertainties(source='sql', assume_inf=False)  # establish mysql connection


#Need blackout times for high-rate noise regions
def inBlackoutTime(time, blackoutTimes):
    for blackouts in blackoutTimes:
        if blackouts[0] < time and time < blackouts[1]:
            return True
    return False

blackoutFile = open('Code/BlackoutCuts.json')
blackoutData = json.load(blackoutFile)
blackoutFile.close()

blackoutTimes = []

for iB, tStart in enumerate(blackoutData['BlackoutCutStarts']):
    tEnd = blackoutData['BlackoutCutEnds'][iB]
    blackoutTimes.append([tStart, tEnd])


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

def getVrms(nurFile, save_chans, station_id, check_forced=False, max_check=1000, plot_avg_trace=False, saveLoc='plots/'):
    template = NuRadioRecoio.NuRadioRecoio(nurFile)


    Vrms_sum = 0
    num_avg = 0
    if plot_avg_trace:
        trace_sum = []

    for i, evt in enumerate(template.get_events()):
        station = evt.get_station(station_id)
        stationtime = station.get_station_time().unix
        if inBlackoutTime(stationtime, blackoutTimes):
            continue

        channelSignalReconstructor.run(evt, station, det)
        for ChId, channel in enumerate(station.iter_channels(use_channels=save_chans)):
            Vrms_sum += channel[chp.noise_rms]
            num_avg += 1
            if plot_avg_trace:
                if trace_sum == []:
                    trace_sum = channel.get_trace()
                else:
                    trace_sum += channel.get_trace()
        
        if num_avg >= max_check:
            break

    if plot_avg_trace:
        plt.plot(trace_sum/num_avg)
        plt.xlabel('sample', label=f'{Vrms_sum/num_avg:.2f} Average Vrms')
        plt.legend()
        plt.savefig(saveLoc + f'stn{station_id}_average_trace.png')

    return Vrms_sum / num_avg

def converter(nurFile, folder, type, save_chans, station_id = 1, blackout=False, det=None, plot=False,
              filter=True, BW=[80*units.MHz, 500*units.MHz], normalize=False, saveTimes=False, timeAdjust=True):
    count = 0
    part = 0
    max_events = 1000000
    ary = np.zeros((max_events, 4, 256))
    if saveTimes:
        art = np.zeros(max_events)
    template = NuRadioRecoio.NuRadioRecoio(nurFile)

#    station_id = 1

    #Normalizing will save traces with values of sigma, rather than voltage
    if normalize:
        Vrms = getVrms(nurFile, save_chans, station_id)
        print(f'normalizing to {Vrms} vrms')



    for i, evt in enumerate(template.get_events()):

        #If in a blackout region, skip event
        station = evt.get_station(station_id)
        stationtime = station.get_station_time().unix
        if inBlackoutTime(stationtime, blackoutTimes):
            continue

        #count = i
        count = i - max_events * part
        if count % 1000 == 0:
            print(f'{count} events processed...')
#        if count % max_events == 0 and not count == 0:
#            print(f'just testing number of events...not saving')
#        continue
        if count >= max_events:
            saveName = f'Code/data/{folder}/{type}_{max_events}events_part{part}.npy'
            print(f'Reached cap, saving events to {saveName}')
            np.save(saveName, ary)
            if saveTimes:
                saveTimes = f'Code/data/{folder}/DateTime_{type}_{max_events}events_part{part}.npy'
                print(f'Saving times to {saveTimes}')
                np.save(saveTimes, art)
            part += 1
            ary = np.zeros((max_events, 4, 256))
            if saveTimes:
                art = np.zeros(max_events)
        station = evt.get_station(station_id)
        i = i - max_events * part
        count = i
        #i = i - max_events * part

        if saveTimes:
            art[i] = stationtime

        #Example of getting date from saved stationtime/testing it works
        if False:
            print(f'Datetime event is ' + datetime.utcfromtimestamp(art[i]).strftime('%Y-%m-%d %H:%M:%S'))

        if timeAdjust:
            triggerTimeAdjuster.run(evt, station, det)  #Run to cut any traces that are too long to be the length expected, based around trigger

        for ChId, channel in enumerate(station.iter_channels(use_channels=save_chans)):

#Shouldn't need these anymore since I resample and length adjust before saving
#            if not channel.get_sampling_rate() == 1:
#                print(f'resampling rate is {channel.get_sampling_rate()}')
#                print(f'resampling')
#                channelResampler.run(evt, station, det, 1*units.GHz)
#            print(f'channel sampling rate is now {channel.get_sampling_rate()}')
#            channelLengthAdjuster.run(evt, station, channel_ids=[channel.get_id()])
            if filter:
                channelBandPassFilter.run(evt, station, det, passband=[BW[0], 1000*units.MHz], filter_type='butter', order=10)
                channelBandPassFilter.run(evt, station, det, passband=[1*units.MHz, BW[1]], filter_type='butter', order=5)
            channelStopFilter.run(evt, station, det, prepend=0*units.ns, append=0*units.ns)

            y = channel.get_trace()
            t = channel.get_times()
            if len(y) > 257:
                channelLengthAdjuster.run(evt, station, channel_ids=[channel.get_id()])
                y = channel.get_trace()
                t = channel.get_times()
#                plt.plot(t, y)
#                plt.title(f'ch {channel.get_id()}')
#                plt.show()
#                print(f'len y {len(y)}')
#                print(f'num of samples{channel.get_number_of_samples()}')
#            continue
            #Array with 3 dimensions
            #Dim 1 = event number
            #Dim 2 = Channel identifier (0-X)
            #Dim 3 = Samples, 256 long, voltage trace
            if normalize:
                y = y / Vrms
            ary[i, ChId] = y

        if plot and i % 1000 == 0:
            plotTrace(ary[i], f"Sim RCR {i}",f"Code/data/{folder}/Sim_RCR_{i}.png")

    ary = ary[0:(count - max_events * part)]
    print(ary.shape)
    """
    max_amp_mask = np.zeros(len(ary), dtype=bool)
    for i, traces in enumerate(ary):
        maxCh = np.max(ary[i])
        if maxCh > 0.7:
            max_amp_mask[i] = True
        else:
            max_amp_mask[i] = False
    ary = ary[max_amp_mask]
    print(ary.shape)
    """
#    np.save(f'DeepLearning/data/2ndpass/{folder}_{len(ary)}events_MaskedMaxAmp_part{part}.npy', ary)
    saveName = f'Code/data/{folder}/{type}_{len(ary)}events_part{part}.npy'
    print(f'Saving to {saveName}')
    np.save(saveName, ary)

    if saveTimes:
        art = art[0:(count - max_events * part)]
        saveTimes = f'Code/data/{folder}/DateTime_{type}_{max_events}events_part{part}.npy'
        np.save(saveTimes, art)
    return

#file = '/Users/astrid/Desktop/st61_deeplearning/data/stn61_2of4trigger_noiseless_processed.nur'
#ReflCrFiles = ['DeepLearning/data/2ndpass/MB_old_100s_Refl_CRs_2500Evts_Noise_True_Amp_True_min0_max500.nur', 'DeepLearning/data/2ndpass/MB_old_100s_Refl_CRs_2500Evts_Noise_True_Amp_True$
#               'DeepLearning/data/2ndpass/MB_old_100s_Refl_CRs_2500Evts_Noise_True_Amp_True_min1000_max1500.nur', 'DeepLearning/data/2ndpass/MB_old_100s_Refl_CRs_2500Evts_Noise_True_Amp_$
folder = "4thpass"
MB_RCR_path = f"Code/data/{folder}/simulatedRCRs/200s/"
series = '200s'     #Alternative is 200s

ReflCrFiles = []
for filename in os.listdir(MB_RCR_path):
#    if filename.endswith('_statDatPak.root.nur'):
#        continue
#    else:
#        DataFiles.append(os.path.join(station14_path, filename))
    if filename.startswith('MB_old'):
        ReflCrFiles.append(os.path.join(MB_RCR_path, filename))
saveChannels = [4, 5, 6, 7]

det = detector.Detector(json_filename=f'Code/Config/gen2_MB_old_{series}_footprint576m_infirn.json')
det.update(datetime.datetime(2018, 10, 1))

converter(ReflCrFiles, folder,'ReflCR', saveChannels, 1, det=det, plot=False, timeAdjust=True)

#quit()

#Neutrino evt conversion
"""
print(f'CR worked fine, Nu starting')

NuFiles = ['StationDataAnalysis/data/N02_SimNu_200s_wNoise_wAmp.nur']
saveChannels = [0, 1, 2, 3]
det_nu = generic_detector.GenericDetector(json_filename=f'StationDataAnalysis/configs/MB_generic_200s_wDipole.json', assume_inf=False, antenna_by_depth=False, default_station=1)
det_nu.update(datetime.datetime(2018, 10, 1))

converter(NuFiles, 'Nu', saveChannels, 1, det_nu)
"""


#Existing data conversion

station14_path = "/pub/ariannaproject/station_nur/station_14/"

DataFiles = []
for filename in os.listdir(station14_path):
    if filename.endswith('_statDatPak.root.nur'):
        continue
    else:
        DataFiles.append(os.path.join(station14_path, filename))

saveChannels = [0, 1, 2, 3]
converter(DataFiles, folder, 'Station14_Data', saveChannels, station_id = 14, blackout=True, plot=False, timeAdjust=False)
