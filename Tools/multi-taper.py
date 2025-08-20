import numpy as np
from scipy import signal
from scipy import fft as sp_fft
from pandas import DataFrame


def mt_cross_spec(Sx, Sy, weights):
    """
    Compute one-sided cross-spectrum between two tapered time-series, derived from a
    multi-taper spectral estimation.
    Sx, Sy: ndarray (K, nf), complex DFTs of the tapered sequence x and y
    weights: ndarray (K, nf) or (K,), weights for different tapers
    """
    nf = Sx.shape[-1]
    if isinstance(weights, (list, tuple)):
        autospectrum = False
        weights_x = weights[0]
        weights_y = weights[1]
        denom = (weights_x**2).sum(axis=0) ** 0.5
        denom *= (weights_y**2).sum(axis=0) ** 0.5
    else:
        autospectrum = True
        weights_x = weights
        weights_y = weights
        denom = (weights**2).sum(axis=0)
    Sxy = ((weights_x*Sx) * (weights_y*Sy).conj()).sum(axis=0) / denom
    # use one-sided spectrum
    iNyq = nf//2 + 1
    one_side = slice(0, iNyq)
    Sxy = Sxy[one_side]
    # double power at duplicated freqs
    # the first & last one (zero & Nyquist) is not paired for even-number fft
    dup_freq = slice(1, -1)
    Sxy[dup_freq] *= 2
    if autospectrum:
        return Sxy.real # imginary part is 0
    return Sxy


def mt_adapt_weights(Sx, eigens, sig2, max_iter=150):
    """
    Find the adaptive weights (two-sided) for K direct spectral estimators of DPSS tapered signals.
    Adapted from the Matlab mutitaper function 'pmtm'
    This version uses the equations from Percival & Walden (1993) pp 368-370
    """
    nf = Sx.shape[-1]
    Sxx = abs(Sx)**2
    # tolerance for acceptance of spectral estimate
    tol = 5e-4*sig2/nf
    a = sig2*(1 - eigens[:,None])
    S1 = np.zeros((1, nf))
    Samt = Sxx[:2,:].mean(axis=0) # initial guess
    # avoid sigularity
    a[a <= 0] = 1e-22
    Samt[Samt == 0] = 1e-64
    # do the iteration
    i = 0
    while abs(Samt-S1).sum() > tol:
        i += 1
        if i > max_iter:
            # warning and return non-converged weights
            e_s = 'Breaking due to iterative meltdown in mt_adaptive_weights.'
            warnings.warn(e_s, RuntimeWarning)
            break
        b = Samt/(Samt*eigens[:,None] + a)
        wk = b**2 * eigens[:,None]
        # update S1 with the weights
        S1 = (wk*Sxx).sum(axis=0) / wk.sum(axis=0)
        Samt, S1 = S1, Samt # swap
    weights = np.sqrt(wk)
    return weights


def mt_psd(x, fs=1, NW=8, low_bias=True, agg='adapt', detrend='linear'):
    """
    Estimate one-sided power spectral density (PSD) using multitaper method.
    Ref: https://github.com/preraulab/multitaper_toolbox/blob/master/python/multitaper_spectrogram_python.py
    x: time series of which PSD is estimated
    fs: sampling frequency in Hz
    NW: (time length)*(half bandwidth in spectral concentration maximization)
    low_bias: use tapers with > 90% energy concentration
    agg: method to combine the PSD estimates of different tapers
    detrend: {False, 'constant', 'linear'}, specify how to detrend data
    """
    Kmax = int(NW*2-1) # number of tapers
    if len(x)%2:
        x = x[1:] # even number of data for fft
    N = len(x)
    if detrend is not False:
        x = signal.detrend(x, type=detrend)
    tapers, eigens = signal.windows.dpss(N, NW=NW, Kmax=Kmax, sym=False, return_ratios=True)
    tapers = tapers/(tapers**2).sum(axis=1).reshape(Kmax, 1) # sym=False gives slightly un-normalized windows
    # tapers has shape (Kmax, N)
    if low_bias:
        keepers = eigens > 0.9
        K = keepers.sum()
        if K < Kmax:
            tapers = tapers[keepers,:]
            eigens = eigens[keepers]
    Sx = sp_fft.fft(tapers*x, n=N, axis=-1)
    if agg == 'unity':
        weights = np.ones((K, 1))
    elif agg == 'eigen':
        weights = np.sqrt(eigens).reshape(K, 1)
    elif agg == 'adapt':
        sig2 = np.var(x)
        weights = mt_adapt_weights(Sx, eigens, sig2)
    else:
        raise ValueError('multitaper weighting method not supported.')
    PSDf = mt_cross_spec(Sx, Sx, weights)
    # don't normalize the spectrum by 1/N as usual, since the taper
    # windows are orthonormal, they effectively scale the signal by 1/N
    PSDf /= fs
    f = sp_fft.rfftfreq(N, d=1/fs)
    return f, PSDf


def mt_csd(x, y, fs=1, NW=8, low_bias=True, agg='adapt', detrend='linear'):
    """
    Estimate one-sided cross-power spectral density (CSD) using multitaper method.
    Ref: https://github.com/preraulab/multitaper_toolbox/blob/master/python/multitaper_spectrogram_python.py
    x, y: time series between which CSD is estimated
    fs: sampling frequency in Hz
    NW: (time length)*(half bandwidth in spectral concentration maximization)
    low_bias: use tapers with > 90% energy concentration
    agg: method to combine the PSD estimates of different tapers
    detrend: {False, 'constant', 'linear'}, specify how to detrend data
    """
    Kmax = int(NW*2-1) # number of tapers
    if len(x)%2:
        x, y = x[1:], y[1:] # even number of data for fft
    N = len(x)
    if detrend is not False:
        x = signal.detrend(x, type=detrend)
        y = signal.detrend(y, type=detrend)
    tapers, eigens = signal.windows.dpss(N, NW=NW, Kmax=Kmax, sym=False, return_ratios=True)
    tapers = tapers/(tapers**2).sum(axis=1).reshape(Kmax, 1) # sym=False gives slightly un-normalized windows
    # tapers has shape (Kmax, N)
    if low_bias:
        keepers = eigens > 0.9
        K = keepers.sum()
        if K < Kmax:
            tapers = tapers[keepers,:]
            eigens = eigens[keepers]
    Sx = sp_fft.fft(tapers*x, n=N, axis=-1)
    Sy = sp_fft.fft(tapers*y, n=N, axis=-1)
    weights = [None]*2
    if agg == 'unity':
        weights[0] = np.ones((K, 1))
        weights[1] = np.ones((K, 1))
    elif agg == 'eigen':
        weights[0] = np.sqrt(eigens).reshape(K, 1)
        weights[1] = np.sqrt(eigens).reshape(K, 1)
    elif agg == 'adapt':
        sig2x, sig2y = np.var(x), np.var(y)
        weights[0] = mt_adapt_weights(Sx, eigens, sig2x)
        weights[1] = mt_adapt_weights(Sy, eigens, sig2y)
    else:
        raise ValueError('multitaper weighting method not supported.')
    CSDf = mt_cross_spec(Sx, Sy, weights)
    CSDf *= 2 # the full CSD has contribution from Syx too, Syx.conj() = Sxy
    # don't normalize the spectrum by 1/N as usual, since the taper
    # windows are orthonormal, they effectively scale the signal by 1/N
    CSDf /= fs
    f = sp_fft.rfftfreq(N, d=1/fs)
    return f, CSDf


def spec_average(omega, spec, nprsv=10, navg=50):
    """
    Average spectrum in frequency bands.
    nprsv: number of frequencies to preserve at the low end.
    navg: number of frequency bands resulted from averaging.
    """
    domega = omega[0]
    prsv_bins = omega[:nprsv] - domega/2
    omega_b = omega[nprsv-1] + domega/2
    omega_e = omega[-1] + domega/2
    avg_bins = np.logspace(np.log10(omega_b), np.log10(omega_e), navg+1)
    omega_bins = np.concatenate((prsv_bins,avg_bins))
    
    df = DataFrame(data=dict(omega=omega, spec=spec))
    dfm = df.groupby(cut(df.omega, omega_bins)).mean().dropna()
    return dfm.omega.values, dfm.spec.values


def get_psd(x, fs=1, method='multitaper', detrend='linear', spec_avg=True):
    """
    Estimate one-sided power spectral density (PSD) for time series x.
    Return PSD in unit of [x]**2/(radian/s)
    fs: sampling frequency in Hz
    method: method for PSD calculation
    """
    if method is 'wosa':
        N = len(x)
        nps = sp_fft.next_fast_len( int(N/4) ) # number of points per segment
        overlap = 0.5 # fraction for overlapping between segments
        nol = int(nps*overlap)
        f, PSDf = signal.welch(x, fs=fs, nperseg=nps, noverlap=nol, detrend=detrend)
    elif method is 'multitaper':
        f, PSDf = mt_psd(x, fs=fs, NW=16, agg='adapt', detrend=detrend)
    else:
        raise ValueError('spectrum method not supported.')
    # convert to radial spectrum and remove zero freq
    omega = 2*pi*f[1:]
    PSD = PSDf[1:]/2/pi
    # average in frequecy bins
    if spec_avg:
        omega, PSD = spec_average(omega, PSD)
    return omega, PSD


def get_csd(x, y, fs=1, method='multitaper', detrend='linear', spec_avg=True):
    """
    Estimate one-sided cross-power spectral density (CSD) for time series x and y.
    Return CSD in unit of [x]**2/(radian/s)
    fs: sampling frequency in Hz
    method: method for CSD calculation
    """
    if y.shape != x.shape:
        raise ValueError('shape mismatch between x, y')
    
    if method is 'wosa':
        N = len(x)
        nps = sp_fft.next_fast_len( int(N/4) ) # number of points per segment
        overlap = 0.5 # fraction for overlapping between segments
        nol = int(nps*overlap)
        f, CSDf = signal.csd(x, y, fs=fs, nperseg=nps, noverlap=nol, detrend=detrend)
    elif method is 'multitaper':
        f, CSDf = mt_csd(x, y, fs=fs, NW=50, agg='adapt', detrend=detrend)
    else:
        raise ValueError('spectrum method not supported.')
    # convert to radial spectrum and remove zero freq
    omega = 2*pi*f[1:]
    CSD = CSDf[1:]/2/pi
    # average in frequecy bins
    if spec_avg:
        omega, CSD = spec_average(omega, CSD)
    return omega, CSD
