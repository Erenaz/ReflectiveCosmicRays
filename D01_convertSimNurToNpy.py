import datetime
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units, fft
from NuRadioReco.modules import channelResampler as CchannelResampler
from NuRadioReco.modules.ARIANNA import hardwareResponseIncorporator as ChardwareResponseIncorporator
from NuRadioReco.modules import channelTimeWindow as cTWindow
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.detector import generic_detector

# Initialize modules
channelResampler = CchannelResampler.channelResampler()
channelResampler.begin(debug=False)
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
hardwareResponseIncorporator = ChardwareResponseIncorporator.hardwareResponseIncorporator()
hardwareResponseIncorporator.begin(debug=False)
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
correlationDirectionFitter.begin(debug=False)
channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()
cTW = cTWindow.channelTimeWindow()
cTW.begin(debug=False)
triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
triggerTimeAdjuster.begin(pre_trigger_time=30 * units.ns)
channelLengthAdjuster = NuRadioReco.modules.channelLengthAdjuster.channelLengthAdjuster()
channelLengthAdjuster.begin()

# Load blackout times
with open('Code/BlackoutCuts.json') as blackoutFile:
    blackoutData = json.load(blackoutFile)
blackoutTimes = list(zip(blackoutData['BlackoutCutStarts'], blackoutData['BlackoutCutEnds']))

def in_blackout_time(time, blackout_times):
    if datetime.datetime.fromtimestamp(time) > datetime.datetime(2019, 3, 31):
        return True
    return any(start < time < end for start, end in blackout_times)

def plot_trace(traces, title, save_loc, show=False):
    fig, axs = plt.subplots(4, 1)
    for ch_id, trace in enumerate(traces):
        axs[ch_id].plot(trace)
    plt.suptitle(title)
    if show:
        plt.show()
    else:
        plt.savefig(save_loc, format='png')
    plt.close(fig)

def plot_traces_and_freq(traces, title, save_loc, sampling_rate=2, show=False):
    x = np.linspace(1, int(256 / sampling_rate), num=256)
    x_freq = np.fft.rfftfreq(len(x), d=(1 / sampling_rate * units.GHz)) / units.MHz

    plt.plot(x, traces[0] * 100, color='orange')
    plt.plot(x, traces[1] * 100, color='blue')
    plt.plot(x, traces[2] * 100, color='purple')
    plt.plot(x, traces[3] * 100, color='green')
    plt.xlabel('time [ns]', fontsize=18)
    plt.ylabel('Amplitude (mV)')
    plt.xlim(-3, 260 / sampling_rate)
    plt.suptitle(title)
    plt.savefig(f'Code/plots/Station_14/NuSearchTraces_{title}.png', format='png')
    plt.clf()

    freqs = [np.abs(fft.time2freq(trace, sampling_rate * units.GHz)) for trace in traces]
    plt.plot(x_freq / 1000, freqs[0], color='orange', label='Channel 0')
    plt.plot(x_freq / 1000, freqs[1], color='blue', label='Channel 1')
    plt.plot(x_freq / 1000, freqs[2], color='purple', label='Channel 2')
    plt.plot(x_freq / 1000, freqs[3], color='green', label='Channel 3')
    plt.xlabel('Frequency [GHz]', fontsize=18)
    plt.ylabel('Amplitude')
    plt.xlim(-0.003, 1.050)
    plt.xticks(size=13)
    plt.suptitle(title)
    plt.savefig(f'Code/plots/Station_14/NuSearchFreqs_{title}.png', format='png')
    plt.clf()

def get_vrms(nur_file, save_chans, station_id, det, check_forced=False, max_check=1000, plot_avg_trace=False, save_loc='plots/'):
    template = NuRadioRecoio.NuRadioRecoio(nur_file)
    vrms_sum, num_avg, trace_sum = 0, 0, []

    for i, evt in enumerate(template.get_events()):
        station = evt.get_station(station_id)
        if in_blackout_time(station.get_station_time().unix, blackoutTimes):
            continue

        channelSignalReconstructor.run(evt, station, det)
        for ch_id, channel in enumerate(station.iter_channels(use_channels=save_chans)):
            vrms_sum += channel[chp.noise_rms]
            num_avg += 1
            if plot_avg_trace:
                trace_sum = trace_sum + channel.get_trace() if trace_sum else channel.get_trace()

        if num_avg >= max_check:
            break

    if plot_avg_trace:
        plt.plot(trace_sum / num_avg)
        plt.xlabel('sample')
        plt.title(f'{vrms_sum / num_avg:.2f} Average Vrms')
        plt.legend()
        plt.savefig(os.path.join(save_loc, f'stn{station_id}_average_trace.png'))

    return vrms_sum / num_avg

def converter(nur_file, folder, data_type, save_chans, station_id=1, det=None, plot=False, blackout=True,
              filter=False, bw=[80 * units.MHz, 500 * units.MHz], normalize=False, save_times=False, time_adjust=True, sim=False, reconstruct=False):
    count, part, max_events = 0, 0, 500000
    ary = np.zeros((max_events, 4, 256))
    if save_times:
        art = np.zeros(max_events)
    if sim:
        arw = np.zeros(max_events)
        arz = np.zeros((max_events, 3))
    template = NuRadioRecoio.NuRadioRecoio(nur_file)

    if normalize:
        vrms = get_vrms(nur_file, save_chans, station_id, det)
        print(f'Normalizing to {vrms} Vrms')

    if reconstruct:
        correlationDirectionFitter.begin(debug=False)
        arr = np.zeros(max_events)

    for i, evt in enumerate(template.get_events()):
        station = evt.get_station(station_id)
        station_time = station.get_station_time().unix

        if blackout and in_blackout_time(station_time, blackoutTimes):
            continue

        if i % 1000 == 0:
            print(f'{i} events processed...')
        if count >= max_events:
            save_data(ary, art, arw, arz, arr, count, max_events, part, folder, data_type, save_times, sim, reconstruct)
            part += 1
            ary = np.zeros((max_events, 4, 256))
        count = i - max_events * part

        if save_times:
            art[count] = station_time
        if sim:
            sim_shower = evt.get_sim_shower(0)
            sim_energy = sim_shower[shp.energy]
            sim_zen = sim_shower[shp.zenith]
            sim_azi = sim_shower[shp.azimuth]
            event_rate = get_cr_event_rate(np.log10(sim_energy), np.rad2deg(sim_zen) * units.deg)
            arw[count] = event_rate
            arz[count] = [np.log10(sim_energy), np.rad2deg(sim_zen), np.rad2deg(sim_azi)]

        if reconstruct:
            correlationDirectionFitter.run(evt, station, det, n_index=1.35, ZenLim=[0 * units.deg, 180 * units.deg])
            zen = station[stnp.zenith]
            arr[count] = np.rad2deg(zen)

        for ch_id, channel in enumerate(station.iter_channels(use_channels=save_chans)):
            if filter:
                channelBandPassFilter.run(evt, station, det, passband=[bw[0], 1000 * units.MHz], filter_type='butter', order=10)
                channelBandPassFilter.run(evt, station, det, passband=[1 * units.MHz, bw[1]], filter_type='butter', order=5)
            channelStopFilter.run(evt, station, det, prepend=0 * units.ns, append=0 * units.ns)
            y = channel.get_trace()
            if len(y) > 257:
                channelLengthAdjuster.run(evt, station, channel_ids=[channel.get_id()])
                y = channel.get_trace()
            if normalize:
                y /= vrms
            ary[count, ch_id] = y

        if plot and i % 1000 == 0:
            plot_trace(ary[count], f"Sim RCR {i}", f"Code/data/{folder}/Sim_RCR_{i}.png")

    ary = ary[:count]
    save_data(ary, art, arw, arz, arr, count, max_events, part, folder, data_type, save_times, sim, reconstruct)

def save_data(ary, art, arw, arz, arr, count, max_events, part, folder, data_type, save_times, sim, reconstruct):
    save_name = f'Code/data/{folder}/{data_type}_{count}events_part{part}.npy'
    print(f'Saving to {save_name}')
    np.save(save_name, ary)

    if save_times:
        save_times_name = f'Code/data/{folder}/DateTime_{data_type}_{count}events_part{part}.npy'
        print(f'Saving times to {save_times_name}')
        np.save(save_times_name, art)

    if sim:
        save_weights_name = f'Code/data/{folder}/SimWeights_{data_type}_{count}events_part{part}.npy'
        save_params_name = f'Code/data/{folder}/SimParams_{data_type}_{count}events_part{part}.npy'
        print(f'Saving times to {save_weights_name}')
        np.save(save_weights_name, arw)
        np.save(save_params_name, arz)

    if reconstruct:
        save_reconstruct_name = f'Code/data/{folder}/SimReconZeniths_{data_type}_{count}events_part{part}.npy'
        np.save(save_reconstruct_name, arr)

folder = "4thpass"
series = '200s'

# Convert simulated data
if False:
    det = generic_detector.GenericDetector(json_filename=f'../../../pub/jingyz34/Arianna/Code/Config/gen2_MB_old_{series}_footprint576m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
    station_files_path = '/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/200s_2.9.24/'
    sim_rcr_files = [os.path.join(station_files_path, filename) for filename in os.listdir(station_files_path)
                     if filename.startswith(f'NewBacklobes_MB_MB_old_{series}_refracted_CRs_10000Evts_Noise_True_Amp_True') and filename.endswith('.nur')]
    save_channels = [4, 5, 6, 7]
    converter(sim_rcr_files, f'{folder}/Station_14', f'SimRCR_{series}_200s_2.9.24', save_channels, station_id=1, det=det, filter=True, save_times=False, plot=False, sim=True, reconstruct=False, blackout=False)

# Convert simulated backlobe data
if False:
    det = generic_detector.GenericDetector(json_filename=f'../../../pub/jingyz34/Arianna/Code/Config/gen2_MB_BacklobeTest_{series}_footprint576m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
    station_files_path = '/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedBacklobe/200s_5.3.24/'
    sim_backlobe_files = [os.path.join(station_files_path, filename) for filename in os.listdir(station_files_path)
                          if filename.startswith(f'Backlobes_') and filename.endswith('.nur')]
    save_channels = [0, 1, 2, 3]
    converter(sim_backlobe_files, f'{folder}/Station_14', f'Backlobe_{series}_200s_5.3.24', save_channels, station_id=1, det=det, filter=True, save_times=False, plot=False, sim=True, reconstruct=False, blackout=False)

# Existing data conversion
station_id = 14
station_path = f"/dfs8/sbarwick_lab/ariannaproject/station_nur/station_{station_id}/"
data_files = [os.path.join(station_path, filename) for filename in os.listdir(station_path) if not filename.endswith('_statDatPak.root.nur')]
save_channels = [0, 1, 2, 3]
converter(data_files, folder, f'FilteredStation{station_id}_Data', save_channels, station_id=station_id, filter=True, save_times=True, plot=False)