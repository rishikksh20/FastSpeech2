import math
import numpy as np
import librosa
import hparams as hp
from scipy.signal import lfilter
import pyworld as pw
import torch
from scipy.signal import get_window
import librosa.util as librosa_util

np.random.seed(hp.seed)

def label_2_float(x, bits) :
    return 2 * x / (2**bits - 1.) - 1.


def float_2_label(x, bits) :
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)

def load_wav(path) :
    return librosa.load(path, sr=hp.sample_rate)[0]


def save_wav(x, path) :
    librosa.output.write_wav(path, x.astype(np.float32), sr=hp.sample_rate)


def split_signal(x) :
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse, fine) :
    return coarse * 256 + fine - 2**15


def encode_16bits(x) :
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)


mel_basis = None

def energy(y):
    # Extract RMS energy
    S = librosa.magphase(stft(y))[0]
    e = librosa.feature.rms(S=S) # (1 , Number of frames)
    return e.squeeze() # (Number of frames) => (654,)

def pitch(y):
    # Extract Pitch/f0 from raw waveform using PyWORLD
    y = y.astype(np.float64)
    '''
    f0_floor : float
        Lower F0 limit in Hz.
        Default: 71.0
    f0_ceil : float
        Upper F0 limit in Hz.
        Default: 800.0
    '''
    f0, timeaxis = pw.dio(y, hp.sample_rate, frame_period=11.6)  # For hop size 256 frame period is 11.6 ms
    return f0 #   (Number of Frames) = (654,)


def linear_to_mel(spectrogram):
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return np.dot(mel_basis, spectrogram)


def build_mel_basis():
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin)


def normalize(S):
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)


def denormalize(S):
    return (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def spectrogram(y):
    D = stft(y)
    S = amp_to_db(np.abs(D)) - hp.ref_level_db
    return normalize(S)


def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)


def stft(y):
    return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)


def pre_emphasis(x):
    return lfilter([1, -hp.preemphasis], [1], x)


def de_emphasis(x):
    return lfilter([1], [1, -hp.preemphasis], x)


def encode_mu_law(x, mu) :
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True) :
    # TODO : get rid of log2 - makes no sense
    if from_labels : y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x

def reconstruct_waveform(mel, n_iter=32):
    """Uses Griffin-Lim phase reconstruction to convert from a normalized
    mel spectrogram back into a waveform."""
    denormalized = denormalize(mel)
    amp_mel = db_to_amp(denormalized)
    S = librosa.feature.inverse.mel_to_stft(
        amp_mel, power=1, sr=hp.sample_rate,
        n_fft=hp.n_fft, fmin=hp.fmin)
    wav = librosa.core.griffinlim(
        S, n_iter=n_iter,
        hop_length=hp.hop_length, win_length=hp.win_length)
    return wav

def quantize_input(input, min, max, num_bins=256):
    
    bins = np.linspace(min, max, num=num_bins)
    quantize = np.digitize(input, bins)
    return quantize


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles).cuda())
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C
#
# def stft(x, n_fft, n_shift, win_length=None, window='hann', center=True,
#          pad_mode='reflect'):
#     # x: [Time, Channel]
#     if x.ndim == 1:
#         single_channel = True
#         # x: [Time] -> [Time, Channel]
#         x = x[:, None]
#     else:
#         single_channel = False
#     x = x.astype(np.float32)
#
#     # FIXME(kamo): librosa.stft can't use multi-channel?
#     # x: [Time, Channel, Freq]
#     x = np.stack([librosa.stft(
#         x[:, ch],
#         n_fft=n_fft,
#         hop_length=n_shift,
#         win_length=win_length,
#         window=window,
#         center=center,
#         pad_mode=pad_mode).T
#         for ch in range(x.shape[1])], axis=1)
#
#     if single_channel:
#         # x: [Time, Channel, Freq] -> [Time, Freq]
#         x = x[:, 0]
#     return x
#
#
# def istft(x, n_shift, win_length=None, window='hann', center=True):
#     # x: [Time, Channel, Freq]
#     if x.ndim == 2:
#         single_channel = True
#         # x: [Time, Freq] -> [Time, Channel, Freq]
#         x = x[:, None, :]
#     else:
#         single_channel = False
#
#     # x: [Time, Channel]
#     x = np.stack([librosa.istft(
#         x[:, ch].T,  # [Time, Freq] -> [Freq, Time]
#         hop_length=n_shift,
#         win_length=win_length,
#         window=window,
#         center=center)
#         for ch in range(x.shape[1])], axis=1)
#
#     if single_channel:
#         # x: [Time, Channel] -> [Time]
#         x = x[:, 0]
#     return x
#
#
# def stft2logmelspectrogram(x_stft, fs, n_mels, n_fft, fmin=None, fmax=None,
#                            eps=1e-10):
#     # x_stft: (Time, Channel, Freq) or (Time, Freq)
#     fmin = 0 if fmin is None else fmin
#     fmax = fs / 2 if fmax is None else fmax
#
#     # spc: (Time, Channel, Freq) or (Time, Freq)
#     spc = np.abs(x_stft)
#     # mel_basis: (Mel_freq, Freq)
#     mel_basis = librosa.filters.mel(fs, n_fft, n_mels, fmin, fmax)
#     # lmspc: (Time, Channel, Mel_freq) or (Time, Mel_freq)
#     lmspc = np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))
#
#     return lmspc
#
#
# def spectrogram(x, n_fft, n_shift, win_length=None, window='hann'):
#     # x: (Time, Channel) -> spc: (Time, Channel, Freq)
#     spc = np.abs(stft(x, n_fft, n_shift, win_length, window=window))
#     return spc
#
#
# def logmelspectrogram(x, fs, n_mels, n_fft, n_shift,
#                       win_length=None, window='hann', fmin=None, fmax=None,
#                       eps=1e-10, pad_mode='reflect'):
#     # stft: (Time, Channel, Freq) or (Time, Freq)
#     x_stft = stft(x, n_fft=n_fft, n_shift=n_shift, win_length=win_length,
#                   window=window, pad_mode=pad_mode)
#
#     return stft2logmelspectrogram(x_stft, fs=fs, n_mels=n_mels, n_fft=n_fft,
#                                   fmin=fmin, fmax=fmax, eps=eps)
#
#
# EPS = 1e-10
#
#
# def logmelspc_to_linearspc(lmspc, fs, n_mels, n_fft, fmin=None, fmax=None):
#     """Convert log Mel filterbank to linear spectrogram.
#
#     Args:
#         lmspc (ndarray): Log Mel filterbank (T, n_mels).
#         fs (int): Sampling frequency.
#         n_mels (int): Number of mel basis.
#         n_fft (int): Number of FFT points.
#         f_min (int, optional): Minimum frequency to analyze.
#         f_max (int, optional): Maximum frequency to analyze.
#
#     Returns:
#         ndarray: Linear spectrogram (T, n_fft // 2 + 1).
#
#     """
#     assert lmspc.shape[1] == n_mels
#     fmin = 0 if fmin is None else fmin
#     fmax = fs / 2 if fmax is None else fmax
#     mspc = np.power(10.0, lmspc)
#     mel_basis = librosa.filters.mel(fs, n_fft, n_mels, fmin, fmax)
#     inv_mel_basis = np.linalg.pinv(mel_basis)
#     spc = np.maximum(EPS, np.dot(inv_mel_basis, mspc.T).T)
#
#     return spc
#
#
# def griffin_lim(spc, n_fft, n_shift, win_length, window='hann', n_iters=100):
#     """Convert linear spectrogram into waveform using Griffin-Lim.
#
#     Args:
#         spc (ndarray): Linear spectrogram (T, n_fft // 2 + 1).
#         n_fft (int): Number of FFT points.
#         n_shift (int): Shift size in points.
#         win_length (int): Window length in points.
#         window (str, optional): Window function type.
#         n_iters (int, optionl): Number of iterations of Griffin-Lim Algorithm.
#
#     Returns:
#         ndarray: Reconstructed waveform (N,).
#
#     """
#     # assert the size of input linear spectrogram
#     assert spc.shape[1] == n_fft // 2 + 1
#
#     spc = np.abs(spc.T)
#     y = librosa.griffinlim(
#         S=spc,
#         n_iter=n_iters,
#         hop_length=n_shift,
#         win_length=win_length,
#         window=window
#     )
#     # else:
#     #     # use slower version of Grriffin-Lim algorithm
#     #     logging.warning("librosa version is old. use slow version of Grriffin-Lim algorithm."
#     #                     "if you want to use fast Griffin-Lim, please update librosa via "
#     #                     "`source ./path.sh && pip install librosa==0.7.0`.")
#     #     cspc = np.abs(spc).astype(np.complex).T
#     #     angles = np.exp(2j * np.pi * np.random.rand(*cspc.shape))
#     #     y = librosa.istft(cspc * angles, n_shift, win_length, window=window)
#     #     for i in range(n_iters):
#     #         angles = np.exp(1j * np.angle(librosa.stft(y, n_fft, n_shift, win_length, window=window)))
#     #         y = librosa.istft(cspc * angles, n_shift, win_length, window=window)
#
#     return y