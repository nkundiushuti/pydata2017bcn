"""
    This file is based on DeepConvSep and librosa

    Copyright (c) 2014-2017 Marius Miron  <miron.marius at gmail.com>

    DeepConvSep is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DeepConvSep is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with DeepConvSep.  If not, see <http://www.gnu.org/licenses/>.
 """
import scipy
import numpy as np
import os
import util

class Transforms(object):
    """
    A general class which can be extended to compute features from audio (STFT,MEL)

    Parameters
    ----------
    frameSize : int, optional
        The frame size for the analysis in samples
    hopSize : int, optional
        The hop size for the analysis in samples
    sampleRate : int, optional
        The sample rate at which to read the signals
    window : function, optional
        The window function for the analysis
    
    """
    def __init__(self, ttype='fft', bins=48, frameSize=1024, hopSize=256, tffmin=0, tffmax=22050, iscale = 'lin', sampleRate=44100, window=np.hanning):
        self.bins = bins
        self.frameSize = frameSize
        self.hopSize = hopSize
        self.fmin = float(tffmin)
        self.fmax = float(tffmax)
        self.iscale = iscale
        self.sampleRate = sampleRate
        self.ttype = ttype
        self.window = window(self.frameSize)
        self.suffix = "" #for backwards compatibility

    def compute_transform(self, audio, out_path=None, save=True,suffix="",sampleRate=None):
        """
        Compute the features for an audio signal.
            The audio signal \"audio\" is a numpy array with the shape (t,i) - t is time and i is the id of signal
            Depending on the variable \"save\", it can save the features to a binary file, accompanied by a shape file,
            which is useful for loading the binary data afterwards
        
        Parameters
        ----------
        audio : 1D numpy array
            The array comprising the audio signals
        out_path : string, optional
            The path of the directory where to save the audio.
        save : bool, optional
            To return or to save in the out_path the computed features
        Yields
        ------
        mag : 2D numpy array
            The features computed for each of the signals in the audio array, e.g. magnitude spectrograms
        """
        self.out_path = out_path
        assert os.path.isdir(os.path.dirname(self.out_path)), "path to save tensor does not exist"

        if sampleRate is not None:
            self.sampleRate = sampleRate
            #self.fmax=float(self.sampleRate)/2

        #compute features 
        mag=self.compute_file(audio, sampleRate=self.sampleRate)
            
        if save and self.out_path is not None:
            util.saveTensor(mag,self.out_path,suffix)
            mag = None
        else:
            return mag

    def compute_file(self,audio):
        #to be extended
        return None


class transformFFT(Transforms):
    """
    A class to help computing the short time Fourier transform (STFT) 
    
    Examples
    --------
    ### 1. Computing the STFT of a matrix of signals \"audio\" and writing the STFT data in \"path\" 
    tt1=transformFFT(frameSize=2048, hopSize=512, sampleRate=44100)
    tt1.compute_transform(audio,out_path=path)

    ### 2. Computing the STFT of a single signal \"audio\" and returning the magnitude 
    tt1=transformFFT(frameSize=2048, hopSize=512, sampleRate=44100)
    mag,ph = tt1.compute_file(audio)
    
    """

    def __init__(self, ttype='fft', bins=48, frameSize=1024, hopSize=256, tffmin=25, tffmax=18000, iscale = 'lin', sampleRate=44100, window=np.hanning):
        super(transformFFT, self).__init__(ttype='fft', bins=bins, frameSize=frameSize, hopSize=hopSize, tffmin=tffmin, tffmax=tffmax, iscale = iscale, sampleRate=sampleRate, window=window)

    def compute_file(self,audio, sampleRate=44100):
        """
        Compute the STFT for an audio signal

        Parameters
        ----------
        audio : 1D numpy array
            The array comprising the audio signals
        sampleRate : int, optional
            The sample rate at which to read the signals
        Yields
        ------
        mag : 2D numpy array
            The features computed for each of the signals in the audio array, e.g. magnitude spectrograms
        """
        X = stft_norm(audio, window=self.window, hopsize=float(self.hopSize), nfft=float(self.frameSize), fs=float(sampleRate))
        mag = np.abs(X)
        mag = mag  / np.sqrt(self.frameSize) #normalization
        
        X = None
        return mag


class transformMEL(Transforms):
    """
    A class to help computing the short time Fourier transform (STFT) 
    
    """

    def __init__(self, ttype='mel', bins=48, frameSize=1024, hopSize=256, tffmin=25, tffmax=18000, iscale = 'lin', sampleRate=44100, window=np.hanning):
        super(transformMEL, self).__init__(ttype='mel', bins=bins, frameSize=frameSize, hopSize=hopSize, tffmin=tffmin, tffmax=tffmax, iscale = iscale, sampleRate=sampleRate, window=window)

    def compute_file(self,audio, sampleRate=44100):
        """
        Compute the MEL spectrogram for an audio signal

        Parameters
        ----------
        audio : 1D numpy array
            The array comprising the audio signals
        sampleRate : int, optional
            The sample rate at which to read the signals
        Yields
        ------
        mag : 2D numpy array
            The features computed for each of the signals in the audio array, e.g. mel spectrograms
        """
        #compute the STFT magnitude spectrogram
        X = stft_norm(audio, window=self.window, hopsize=float(self.hopSize), nfft=float(self.frameSize), fs=float(sampleRate))
        mag = np.abs(X)
        mag = mag  / np.sqrt(self.frameSize) #normalization
        X = None

        #compute the mel filters using librosa
        mel_basis = mel(self.sampleRate, n_fft = float(self.frameSize), n_mels=self.bins, fmin=self.fmin, fmax=self.fmax, htk=False,norm=1)
        
        return np.dot(mel_basis, mag.T).T


def sinebell(lengthWindow):
    """
    window = sinebell(lengthWindow)
    
    Computes a \"sinebell\" window function of length L=lengthWindow
    
    The formula is:

    .. math::
    
        window(t) = sin(\pi \\frac{t}{L}), t=0..L-1
        
    """
    window = np.sin((np.pi*(np.arange(lengthWindow)))/(1.0*lengthWindow))
    return window


def stft_norm(data, window=sinebell(2048),
         hopsize=256.0, nfft=2048.0, fs=44100.0):
    """
    X = stft_norm(data,window=sinebell(2048),hopsize=1024.0,
                   nfft=2048.0,fs=44100)
                   
    Computes the short time Fourier transform (STFT) of data.
    
    Inputs:
        data                  :
            one-dimensional time-series to be analyzed
        window=sinebell(2048) :
            analysis window
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)
        fs=44100.0            :
            sampling rate of the signal
        
    Outputs:
        X                     :
            STFT of data
    """
    
    # window defines the size of the analysis windows
    lengthWindow = window.size
    
    lengthData = data.size
    
    # should be the number of frames by YAAFE:
    numberFrames = int(np.ceil(lengthData / np.double(hopsize)) + 2)
    # to ensure that the data array s big enough,
    # assuming the first frame is centered on first sample:
    newLengthData = int((numberFrames-1) * hopsize + lengthWindow)
    
    # !!! adding zeros to the beginning of data, such that the first window is
    # centered on the first sample of data
    data = np.concatenate((np.zeros(int(lengthWindow/2.0)), data))
    
    # zero-padding data such that it holds an exact number of frames
    data = np.concatenate((data, np.zeros(newLengthData - data.size)))
    
    # the output STFT has nfft/2+1 rows. Note that nfft has to be an even
    # number (and a power of 2 for the fft to be fast)
    numberFrequencies = int(nfft / 2 + 1)
    
    STFT = np.zeros([numberFrequencies, numberFrames], dtype=complex)
    
    # storing FT of each frame in STFT:
    for n in np.arange(numberFrames):
        beginFrame = int(n*hopsize)
        endFrame = beginFrame+lengthWindow
        frameToProcess = window*data[beginFrame:endFrame]
        STFT[:,n] = np.fft.rfft(frameToProcess, np.int32(nfft))
        frameToProcess = None
    
    return STFT.T



#imported from librosa
def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
        norm=1):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal

    n_fft     : int > 0 [scalar]
        number of FFT components

    n_mels    : int > 0 [scalar]
        number of Mel bands to generate

    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)

    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use `fmax = sr / 2.0`

    htk       : bool [scalar]
        use HTK formula instead of Slaney

    norm : {None, 1, np.inf} [scalar]
        if 1, divide the triangular mel weights by the width of the mel band 
        (area normalization).  Otherwise, leave all the triangles aiming for 
        a peak value of 1.0

    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    >>> melfb = librosa.filters.mel(22050, 2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])


    Clip the maximum frequency to 8KHz

    >>> librosa.filters.mel(22050, 2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])


    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(melfb, x_axis='linear')
    >>> plt.ylabel('Mel filter')
    >>> plt.title('Mel filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    """

    if fmax is None:
        fmax = float(sr) / 2

    if norm is not None and norm != 1 and norm != np.inf:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = np.linspace(0,float(sr) / 2,int(1 + n_fft//2),endpoint=True)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        print('Empty filters detected in mel frequency basis. '
                      'Some channels will produce empty responses. '
                      'Try increasing your sampling rate (and fmax) or '
                      'reducing n_mels.')

    return weights

def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    """Compute the center frequencies of mel bands.

    Parameters
    ----------
    n_mels    : int > 0 [scalar]
        number of Mel bins

    fmin      : float >= 0 [scalar]
        minimum frequency (Hz)

    fmax      : float >= 0 [scalar]
        maximum frequency (Hz)

    htk       : bool
        use HTK formula instead of Slaney

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        vector of n_mels frequencies in Hz which are uniformly spaced on the Mel
        axis.

    Examples
    --------
    >>> librosa.mel_frequencies(n_mels=40)
    array([     0.   ,     85.317,    170.635,    255.952,
              341.269,    426.586,    511.904,    597.221,
              682.538,    767.855,    853.173,    938.49 ,
             1024.856,   1119.114,   1222.042,   1334.436,
             1457.167,   1591.187,   1737.532,   1897.337,
             2071.84 ,   2262.393,   2470.47 ,   2697.686,
             2945.799,   3216.731,   3512.582,   3835.643,
             4188.417,   4573.636,   4994.285,   5453.621,
             5955.205,   6502.92 ,   7101.009,   7754.107,
             8467.272,   9246.028,  10096.408,  11025.   ])

    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)

def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels

    Examples
    --------
    >>> librosa.hz_to_mel(60)
    array([ 0.9])
    >>> librosa.hz_to_mel([110, 220, 440])
    array([ 1.65,  3.3 ,  6.6 ])

    Parameters
    ----------
    frequencies   : np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk           : bool
        use HTK formula instead of Slaney

    Returns
    -------
    mels        : np.ndarray [shape=(n,)]
        input frequencies in Mels

    See Also
    --------
    mel_to_hz
    """

    frequencies = np.atleast_1d(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    log_t = (frequencies >= min_log_hz)
    mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies

    Examples
    --------
    >>> librosa.mel_to_hz(3)
    array([ 200.])

    >>> librosa.mel_to_hz([1,2,3,4,5])
    array([  66.667,  133.333,  200.   ,  266.667,  333.333])

    Parameters
    ----------
    mels          : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk           : bool
        use HTK formula instead of Slaney

    Returns
    -------
    frequencies   : np.ndarray [shape=(n,)]
        input mels in Hz

    See Also
    --------
    hz_to_mel
    """

    mels = np.atleast_1d(mels)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region
    log_t = (mels >= min_log_mel)

    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs

  
