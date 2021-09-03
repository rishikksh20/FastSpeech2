import math
import numpy as np
import librosa
from scipy.signal import lfilter
import pyworld as pw
import torch
from scipy.signal import get_window
import librosa.util as librosa_util


def label_2_float(x, bits):
    return 2 * x / (2 ** bits - 1.0) - 1.0


def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.0) * (2 ** bits - 1) / 2
    return x.clip(0, 2 ** bits - 1)


def load_wav(path, hp):
    return librosa.load(path, sr=hp.audio.sample_rate)[0]


def save_wav(x, path, hp):
    librosa.output.write_wav(path, x.astype(np.float32), sr=hp.audio.sample_rate)


def split_signal(x):
    unsigned = x + 2 ** 15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse, fine):
    return coarse * 256 + fine - 2 ** 15


def encode_16bits(x):
    return np.clip(x * 2 ** 15, -(2 ** 15), 2 ** 15 - 1).astype(np.int16)


mel_basis = None


def energy(y):
    # Extract energy
    S = librosa.magphase(stft(y))[0]
    e = np.sqrt(np.sum(S ** 2, axis=0))  # np.linalg.norm(S, axis=0)
    return e.squeeze()  # (Number of frames) => (654,)


def pitch(y, hp):
    # Extract Pitch/f0 from raw waveform using PyWORLD
    y = y.astype(np.float64)
    """
    f0_floor : float
        Lower F0 limit in Hz.
        Default: 71.0
    f0_ceil : float
        Upper F0 limit in Hz.
        Default: 800.0
    """
    f0, timeaxis = pw.dio(
        y,
        hp.audio.sample_rate,
        frame_period=hp.audio.hop_length / hp.audio.sample_rate * 1000,
    )  # For hop size 256 frame period is 11.6 ms
    return f0  # (Number of Frames) = (654,)


def linear_to_mel(spectrogram, hp):
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis(hp)
    return np.dot(mel_basis, spectrogram)


def build_mel_basis(hp):
    return librosa.filters.mel(
        hp.audio.sample_rate,
        hp.audio.n_fft,
        n_mels=hp.audio.num_mels,
        fmin=hp.audio.fmin,
    )


def normalize(S, hp):
    return np.clip((S - hp.audio.min_level_db) / -hp.audio.min_level_db, 0, 1)


def denormalize(S, hp):
    return (np.clip(S, 0, 1) * -hp.audio.min_level_db) + hp.audio.min_level_db


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def spectrogram(y, hp):
    D = stft(y, hp)
    S = amp_to_db(np.abs(D)) - hp.audio.ref_level_db
    return normalize(S, hp)


def melspectrogram(y, hp):
    D = stft(y, hp)
    S = amp_to_db(linear_to_mel(np.abs(D), hp))
    return normalize(S, hp)


def stft(y, hp):
    return librosa.stft(
        y=y,
        n_fft=hp.audio.n_fft,
        hop_length=hp.audio.hop_length,
        win_length=hp.audio.win_length,
    )


def pre_emphasis(x, hp):
    return lfilter([1, -hp.audio.preemphasis], [1], x)


def de_emphasis(x, hp):
    return lfilter([1], [1, -hp.audio.preemphasis], x)


def encode_mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True):
    # TODO : get rid of log2 - makes no sense
    if from_labels:
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def reconstruct_waveform(mel, hp, n_iter=32):
    """Uses Griffin-Lim phase reconstruction to convert from a normalized
    mel spectrogram back into a waveform."""
    denormalized = denormalize(mel)
    amp_mel = db_to_amp(denormalized)
    S = librosa.feature.inverse.mel_to_stft(
        amp_mel,
        power=1,
        sr=hp.audio.sample_rate,
        n_fft=hp.audio.n_fft,
        fmin=hp.audio.fmin,
    )
    wav = librosa.core.griffinlim(
        S, n_iter=n_iter, hop_length=hp.audio.hop_length, win_length=hp.audio.win_length
    )
    return wav


def quantize_input(input, min, max, num_bins=256):
    bins = np.linspace(min, max, num=num_bins)
    quantize = np.digitize(input, bins)
    return quantize


def window_sumsquare(
    window,
    n_frames,
    hop_length=200,
    win_length=800,
    n_fft=800,
    dtype=np.float32,
    norm=None,
):
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
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
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
