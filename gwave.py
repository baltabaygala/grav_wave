import gwosc
import gwpy
import matplotlib
import pylab
import numpy as np
from pycbc.waveform import get_td_waveform
import matplotlib.pyplot as plt
from pycbc.catalog import Merger
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import matched_filter
from pycbc.filter import sigma

apx = 'IMRPhenomD'

def simulate_gw(sample_rate, data_length, m_1, m_2, apx = 'IMRPhenomD'):
    data = np.random.normal(size=[sample_rate * data_length])
    times = np.arange(len(data)) / float(sample_rate)
    hp1, _ = get_td_waveform(approximant=apx,
                             mass1=m_1,
                             mass2=m_2,
                             delta_t=1.0 / sample_rate,
                             f_lower=25)
    # Cross-correlation of the signal with white noise as a signal-to-noise ratio
    hp1 = hp1 / max(np.correlate(hp1, hp1, mode='full')) ** 0.5
    # Shift the waveform to start at a random time in the Gaussian noise data.
    waveform_start = np.random.randint(0, len(data) - len(hp1))
    data[waveform_start:waveform_start + len(hp1)] += 10 * hp1.numpy()


    plt.figure()
    plt.title("Signal in the data")
    plt.plot(hp1.sample_times, data[waveform_start:waveform_start + len(hp1)])
    plt.plot(hp1.sample_times, 10 * hp1)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized amplitude')

    plt.savefig('plot_image_GENGW.png')

    return "Finished"

def cross_correlate(sample_rate, data_length,m_1,m_2):
    data = np.random.normal(size=[sample_rate * data_length])
    times = np.arange(len(data)) / float(sample_rate)
    hp1, _ = get_td_waveform(approximant=apx,
                             mass1=m_1,
                             mass2=m_2,
                             delta_t=1.0 / sample_rate,
                             f_lower=25)
    cross_correlation = np.zeros([len(data) - len(hp1)])
    # Cross-correlation of the signal with white noise as a signal-to-noise ratio
    hp1 = hp1 / max(np.correlate(hp1, hp1, mode='full')) ** 0.5
    # Shift the waveform to start at a random time in the Gaussian noise data.
    waveform_start = np.random.randint(0, len(data) - len(hp1))
    data[waveform_start:waveform_start + len(hp1)] += 10 * hp1.numpy()
    hp1_numpy = hp1.numpy()
    for i in range(len(data) - len(hp1_numpy)):
        cross_correlation[i] = (hp1_numpy * data[i:i + len(hp1_numpy)]).sum()

    # plot the cross-correlated data vs time. Superimpose the location of the end of the signal;
    # this is where we should find a peak in the cross-correlation.
    pylab.figure()
    times = np.arange(len(data) - len(hp1_numpy)) / float(sample_rate)
    pylab.plot(times, cross_correlation)
    pylab.plot([waveform_start / float(sample_rate), waveform_start / float(sample_rate)], [-10, 10], 'r:')
    pylab.xlabel('Time (s)')
    pylab.ylabel('Cross-correlation')
    pylab.savefig('plot_image_CC')
    return True
#precond data cond = true return merger, conditioned
def precond_data(merge, st, cond):
    merger = Merger(merge)
    strain =merger.strain(st)
    strain = highpass(strain, 15.0)
    strain = resample_to_delta_t(strain, 1.0 / 2048)
    if cond == True:
        conditioned = strain.crop(2, 2)
        plt.plot(conditioned.sample_times, conditioned)
        plt.xlabel('Time (s)')
        plt.savefig('plot_image_PreCond')
        return merger, conditioned
    else:
        plt.plot(strain.sample_times, strain)
        plt.xlabel('Time (s)')
        plt.savefig('plot_image_PreData')
        return merger, strain
# return psd
def psd(conditioned):
    # We use 4 second samples of our time series in Welch method.
    psd = conditioned.psd(4)
    # Now that we have the psd we need to interpolate it to match our data
    # and then limit the filter length of 1 / PSD. After this, we can
    # directly use this PSD to filter the data in a controlled manner
    psd = interpolate(psd, conditioned.delta_f)
    # 1/PSD will now act as a filter with an effective length of 4 seconds
    # Since the data has been highpassed above 15 Hz, and will have low values
    # below this we need to inform the function to not include frequencies
    # below this frequency.
    psd = inverse_spectrum_truncation(psd, int(4 * conditioned.sample_rate),
                                      low_frequency_cutoff=15)
    return psd
#models wave from collision of m1 and m2. Return template
def model_wave(m_1,m_2,conditioned):
    hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                             mass1=m_1,
                             mass2=m_2,
                             delta_t=conditioned.delta_t,
                             f_lower=20)

    # Resize the vector to match our data
    hp.resize(len(conditioned))
    # Shift the hp
    return hp.cyclic_time_shift(hp.start_time)
# saves plot with peak, return snr, time
def peak_finder(template, conditioned,psd):
    snr = matched_filter(template, conditioned, psd=psd, low_frequency_cutoff=20)
    snr = snr.crop(4 + 4, 4)
    peak = abs(snr).numpy().argmax()
    snrp = snr[peak]
    time = snr.sample_times[peak]
    pylab.figure(figsize=[10, 4])
    pylab.plot(snr.sample_times, abs(snr))
    pylab.ylabel('Signal-to-noise')
    pylab.xlabel('Time (s)')
    pylab.savefig('plot_image_PEAK')

    print("We found a signal at {}s with SNR {}".format(time, abs(snrp)))
    return snr, time, snrp
#return white_data,aligned
def alignment(conditioned, template,time,psd,snrp,merger):
    # Shift the template to the peak time
    dt = time - conditioned.start_time
    aligned = template.cyclic_time_shift(dt)

    # scale the template so that it would have SNR 1 in this data
    aligned /= sigma(aligned, psd=psd, low_frequency_cutoff=20.0)
    # Scale the template amplitude and phase to the peak value
    aligned = (aligned.to_frequencyseries() * snrp).to_timeseries()
    aligned.start_time = conditioned.start_time
    # We do it this way so that we can whiten both the template and the data
    white_data = (conditioned.to_frequencyseries() / psd ** 0.5).to_timeseries()
    white_template = (aligned.to_frequencyseries() / psd ** 0.5).to_timeseries()

    white_data = white_data.highpass_fir(30., 512).lowpass_fir(300, 512)
    white_template = white_template.highpass_fir(30, 512).lowpass_fir(300, 512)

    # Select the time around the merger
    white_data = white_data.time_slice(merger.time - .2, merger.time + .1)
    white_template = white_template.time_slice(merger.time - .2, merger.time + .1)

    pylab.figure(figsize=[15, 3])
    pylab.plot(white_data.sample_times, white_data, label="Data")
    pylab.plot(white_template.sample_times, white_template, label="Template")
    pylab.legend()
    pylab.savefig('plot_image_Overlap')
    return white_data, aligned

def substract(conditioned, aligned, merger):
    subtracted = conditioned - aligned

    # Plot the original data and the subtracted signal data

    for data, title in [(conditioned, 'Original H1 Data'),
                        (subtracted, 'Signal Subtracted from H1 Data')]:
        t, f, p = data.whiten(4, 4).qtransform(.001, logfsteps=100, qrange=(8, 8), frange=(20, 512))
        pylab.figure(figsize=[15, 3])
        pylab.title(title)
        pylab.pcolormesh(t, f, p ** 0.5, vmin=1, vmax=6, shading='auto')
        pylab.yscale('log')
        pylab.xlabel('Time (s)')
        pylab.ylabel('Frequency (Hz)')
        pylab.xlim(merger.time - 2, merger.time + 1)
        pylab.savefig(title)

    return True



