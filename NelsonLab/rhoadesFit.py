import numpy as np
import scipy.io
import scipy.optimize
import scipy.signal
from tkinter import filedialog as fd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def getD2(x, df2):#x is np.ndarray to differentiate; df2 is the square of the step size between successive values in x
    d2x = np.ndarray(x.shape[-1] - 2)
    for j in range(d2x.shape[-1]):
        d2x[j] = (x[j] - 2*x[j+1] + x[j+2])/df2
    return d2x

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]

def findPeaks(raw, srate, min_peak_height=1, min_peak_length=0.1):#raw is EEG signal; srate is sampling frequency
    N = raw.shape[-1]
    freqs = np.fft.rfftfreq(N, 1/srate)
    df = srate/N
    r = autocorr(raw)
    S_r = np.abs(np.fft.rfft(r))
    d2S_r = getD2(S_r, df**2)#get second derivative of |FFT(autocorr(raw))|
    H = np.abs(scipy.signal.hilbert(d2S_r))
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
    # print('Peaks: ' + str(freqs[peaks]))
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

def get_studyid_from_filename(filename):
    if len(filename.split('.')) == 3:
        return filename.split('EEG')[0][3:]
    elif len(filename.split('.')) == 4:
        return filename.split('EEG')[0][3:]
    elif len(filename.split('.')) == 5:
        return filename.split('EEG')[0][3:]
    elif len(filename.split('.')) == 8:
        return filename.split('.')[1]
    else:
        raise AttributeError('Unexpected filename format: {}'.format(filename))

def get_age_from_filename(filename):
    if len(filename.split('.')) == 3:
        return filename.split('EEG')[1].split('.')[0].strip('m')
    elif len(filename.split('.')) == 4:
        return filename.split('EEG')[1].split('.')[0].strip('m')
    elif len(filename.split('.')) == 5:
        return filename.split('EEG')[1].split('.')[0].strip('m')
    elif len(filename.split('.')) == 8:
        return filename.split('.')[3].strip('m')
    else:
        raise AttributeError('Unexpected filename format: {}'.format(filename))

def fitCost(params, args):
    PSD, freqs = args
    bg = expo_function(freqs, *params[:3])
    peaks = gaussian_function(freqs, *params[3:])#consider fixing peak centers
    costVec = (PSD - (bg + peaks))#error
    cost = np.dot(costVec, costVec.transpose())#squared error
    if (PSD - bg < 0).any():#make cost huge for ill-fit background curve
        cost *= 10
    return cost

def clusterPeaks(peaks):
    ks = list()
    allPeaks = []
    for thesePeaks in peaks:
        allPeaks = np.append(allPeaks, thesePeaks)
        ks.append(len(thesePeaks))
    k = int(np.mean(ks))
    print('Finding ' + str(k) + ' peaks...')
    clusters = KMeans(n_clusters=k).fit(allPeaks.reshape(-1, 1))
    print('Found peaks.')
    return clusters.cluster_centers_

def getPSDfit(raws, PSD, freqs, srate, min_peak_height=1, min_peak_length=0.1):
    allPeaks = list()
    for raw in raws:
        peaks, hilbert, pfreqs = findPeaks(raw, srate, min_peak_height=min_peak_height, min_peak_length=min_peak_length)
        allPeaks.append(pfreqs[peaks])
    peakFreqs = clusterPeaks(allPeaks)
    #params/x0 = offset, knee, exp, ctr#, hgt#, wid#, ....., ctrN, hgtN, widN for N peaks
    x0 = [0, freqs.max()/2, 1.5]
    for peak in peakFreqs:
        x0 = np.append(x0, [peak[0], 2*min_peak_height, 2*min_peak_length], axis=0)
    print('Fitting ' + str(len(x0)) + ' parameters...')
    params = scipy.optimize.minimize(fitCost, x0, [PSD, freqs])
    print('Fit as a fiddle.')
    return params, len(peakFreqs)

def loadEEG(fname=None):
    #get raw EEG
    if fname == None:
        print('Select EEG file')
        data = scipy.io.loadmat(fd.askopenfilename(title='Select EEG file'))
    else:
        data = scipy.io.loadmat(fname)
    raws = data['eeg'][0,0]
    bad_chans = data['file_proc_info'][0,0][0][0,0]
    fname = data['file_proc_info'][0,0][1][0,0][0]
    indxs = data['file_proc_info'][0,0][2][0,0]
    srate = data['file_proc_info'][0,0][5][0,0]
    return raws, bad_chans, fname, indxs, srate

def loadPSDs(studyID, age, fname=None):
    if fname == None:
        print('Select PSD file for StudyID ' + str(studyID) + ' at ' + str(age) + 'months')
        data = scipy.io.loadmat(fd.askopenfilename(title='Select PSD file for StudyID ' + str(studyID) + ' at ' + str(age) + 'months'))
    else:
        data = scipy.io.loadmat(fname)
    PSDs = data['eeg_wfp'][0,0].mean(2)
    freqs = data['f'][0,0]
    return PSDs, freqs

def plotFit(data):
    freqs = data['freqs']
    bg = expo_function(freqs, *data['bg_params'])
    peaks = gaussian_function(freqs, *data['peak_params'])
    PSD = data['PSD']
    fig = plt.figure()
    plt.title('Spectrum fit for #' + str(data['studyID']) + ' at ' + str(data['age']) + 'months: ' + data['spec_name'] + '_average')
    plt.plot(freqs, np.log(PSD), 'k', label='Original Spectrum', linewidth=2)
    plt.plot(freqs, np.log(bg), 'b--', label='Aperiodic Fit', linewidth=2)
    plt.plot(freqs, np.log(bg + peaks), 'r', label='Full Model Fit', linewidth=2)
    plt.xlabel('Frequency')
    plt.ylabel('log(Power)')
    plt.legend()
    plt.show()
    return

def rhoadesFit(eegName=None, psdName=None, channels=None):
    raws, bad_chans, fname, indxs, srate = loadEEG(fname=eegName)
    studyID = get_studyid_from_filename(fname)
    age = get_age_from_filename(fname)
    PSDs, freqs = loadPSDs(studyID, age, fname=psdName)
    nElect = raws.shape[0] - 1
    if channels == None:
        channels = 'frontal'
    if isinstance(channels, str):
        spec_name = channels
        if channels == 'frontal':
            if nElect == 128:
                channels = [24, 124, 13, 112, 11, 28, 117]
            elif nElect == 64:
                channels = [13, 62, 3, 8, 16, 57]
        elif channels == 'temporal':
            if nElect == 128:
                channels = [45, 41, 47, 52, 108, 103, 98, 92]
            elif nElect == 64:
                channels = [24, 21, 25, 28, 52, 53, 50, 46]
        elif channels == 'whole':
            if nElect == 128:
                channels = [33, 24, 124, 122, 36, 104, 58, 52, 62, 92, 96, 70, 83, 45, 108]
            elif nElect == 64:
                channels = [15, 13, 62, 61, 17, 54, 27, 28, 34, 46, 49, 37, 40, 24, 52]
        elif channels == 'whole2':
            if nElect == 128:
                channels = [33, 22, 9, 122, 28, 24, 19, 11, 4, 124, 117, 13, 112, 45, 41, 36, 37, 55, 87,
                            104, 103, 108, 47, 52, 67, 62, 77, 92, 98, 58, 65, 70, 75, 83, 90, 96]
            elif nElect == 64:
                channels = [15, 11, 6, 61, 16, 13, 8, 3, 62, 57, 9, 58, 24, 21, 17, 18, 30, 43,
                            54, 53, 52, 25, 28, 33, 34, 41, 46, 50, 27, 32, 37, 40, 45, 49]
        else:
            print('Unexpected channel selection. Please select from the following: frontal, temporal, whole, or whole2')
            return
    else:
        spec_name = 'unnamed'
    selectChans = list()
    [selectChans.append(chan-1) for chan in channels if all(chan != bad_chans) and any(chan == indxs)]
    PSD = PSDs[selectChans, :].mean(0)
    try:
        fitResult, peakNum = getPSDfit(raws[selectChans, :], PSD, freqs, srate, min_peak_length=.5)
    except Exception as e:
        print('Fit failed: ' + str(e))
        data = dict()
        data['PSD'] = PSD
        data['raws'] = raws[selectChans, :]
        data['freqs'] = freqs
        data['srate'] = srate
        return data
    data = dict()
    data['PSD'] = PSD
    data['raws'] = raws[selectChans, :]
    data['freqs'] = np.squeeze(freqs)
    data['srate'] = srate
    data['bg_params'] = fitResult['x'][:3]
    data['peak_params'] = fitResult['x'][3:]
    data['fitResult'] = fitResult
    data['peak_num'] = peakNum
    data['inds_used'] = selectChans
    data['studyID'] = studyID
    data['age'] = age
    data['spec_name'] = spec_name
    plotFit(data)
    return data
