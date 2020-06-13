import torch
import librosa
import numpy as np
import hparams as hp
import torch.nn.functional as F


def calculate_energy(y):
    # Extract RMS energy
    S = librosa.magphase(librosa.core.stft(y, n_fft=1024, hop_length=256))[0]
    e = librosa.feature.rms(S=S)
    return energy_to_one_hot(e)

def retreive_energy(file):

    e= np.load(file)
    return energy_to_one_hot(e)

def energy_to_one_hot(e):

    bins = np.linspace(e.min(), e.max(), num=256)
    e_quantize = np.digitize(e, bins)
    e_quantize = torch.from_numpy(e_quantize).float().to(torch.device("cuda" if hp.ngpu > 0 else "cpu"))
    return F.one_hot(e_quantize, 256).float()