import numpy as np
import pandas as pd
import scipy
from tkinter import filedialog as fd


def getD2(x, df2):#x is np.ndarray to differentiate; df2 is the square of the step size between successive values in x
    d2x = np.ndarray(x.shape[-1] - 2)
    for j in range(d2x.shape[-1]):
        d2x[j] = (x[j] - 2*x[j+1] + x[j+2])/df2
    return d2x

def findPeaks(raw, fs, min_peak_height=1, min_peak_length=0.1):#raw is pandas.Series object; fs is sampling frequency
    N = raw.shape[-1]
    df = fs/N
    r = raw.autocorr()
    S_r = np.abs(scipy.fft.rfft(r))
    d2S_r = getD2(S_r, df**2)#get second derivative of |FFT(autocorr(raw))|
    H = scipy.signal.hilbert(d2S_r)
    #start getting indices of peaks above thresholds:
    thrInds = H > np.nanstd(H)*min_peak_height#get indices above min_peak_height
    #initialize loop variables:
    isPeak = False
    thisPeakLength = 0
    tempInds = thrInds.copy()
    #
    for i, ind in enumerate(thrInds):#erase indices for peaks shorter than min_peak_length
        if ind:
            if isPeak:
                thisPeakLength += 1
            else:#detect peak start
                peakStart = i
                isPeak = True
        else:
            if isPeak:#detect peak end
                if thisPeakLength*df < min_peak_length:#test peak length
                    tempInds[peakStart:i] = False#erase indices of short peaks
                thisPeakLength = 0#reset variables
                isPeak = False
    thrInds = tempInds
    H_thr = np.zeros_like(H)
    H_thr[thrInds] = H[thrInds]
    peaks, peak_props = scipy.signal.find_peaks(H_thr)#find indices of peaks
    peaks -= 2
    freqs = (range(N)+1)*df#DOUBLE CHECK THIS
    """
    "The peak frequency (Pf) is then computed as the position or index on the frequency axis
    corresponding to the peak value shifted by two positions to the left since the envelope is
    computed on the second order derivative or difference function and is therefore shifted
    by two data points." -"Characterizing Peaks in the EEG Power Spectrum" by Sapien Labs
    """
    return peaks, H_thr, freqs

def gaussian_function(xs, *params):
    """Gaussian fitting function.
    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define gaussian function.
    Returns
    -------
    ys : 1d array
        Output values for gaussian function.
    """
    ys = np.zeros_like(xs)
    for ii in range(0, len(params), 3):
        ctr, hgt, wid = params[ii:ii+3]
        ys += hgt * np.exp(-(xs-ctr)**2 / (2*wid**2))
    return ys

def expo_function(xs, *params):
    """Exponential fitting function, for fitting aperiodic component with a 'knee'.
    NOTE: this function requires linear frequency (not log).
    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters (offset, knee, exp) that define Lorentzian function:
        y = 10^offset * (1/(knee + x^exp))
    Returns
    -------
    ys : 1d array
        Output values for exponential function.
    """
    ys = np.zeros_like(xs)
    offset, knee, exp = params
    ys += offset - np.log10(knee + xs**exp)
    return ys

def fitCost(params, PSD, freqs):
    bg = expo_function(freqs, params[0:2])
    peaks = gaussian_function(freqs, params[3:])#consider fixing peak centers
    cost = (PSD - (bg + peaks))**2#squared error
    if any(PSD - bg < 0):#make cost huge for ill-fit background curve
        cost *= 10
    return cost

def getPSDfit(raw, PSD, fs, min_peak_height=1, min_peak_length=0.1):
    # freqs, PSD = scipy.signal.welch(x, fs=fs, window='hamming', nperseg=raw.shape[-1]//8,
    #                         scaling='spectrum', average='median')#median is more resistant to outliers/artifacts than mean
    peaks, hilbert, freqs = findPeaks(raw, fs, min_peak_height=1, min_peak_length=0.1)
    peakFreqs = freqs[peaks]
    #params/x0 = offset, knee, exp, ctr#, hgt#, wid#, ....., ctrN, hgtN, widN for N peaks
    x0 = [0, max(freqs)/2, 1.5]
    for peak in peakFreqs:
        x0 = np.append(x0, [peak, 2*min_peak_length, 2*min_peak_length], axis=0)
    params = scipy.optimize.minimize(fitCost, x0, [PSD, freqs])

def loadData():
    #get raw EEG
    raw = scipy.io.loadmat(fd.askopenfilename(title='Select EEG file'))
