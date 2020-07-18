import torch
import librosa
import numpy as np
import hparams as hp
import torch.nn.functional as F
import os


def calculate_energy(y):
    # Extract RMS energy
    S = librosa.magphase(librosa.core.stft(y, n_fft=1024, hop_length=256))[0]
    e = librosa.feature.rms(S=S)
    return energy_to_one_hot(e)

def retreive_energy(file):

    e= np.load(file)
    return energy_to_one_hot(e)

def de_norm_mean_std(x, mean, std):
    zero_idxs = torch.where(x == 0.0)
    x = x * std + mean
    x[zero_idxs] = 0.0
    return x

def energy_to_one_hot(e, is_training = True):

    # e = de_norm_mean_std(e, hp.e_mean, hp.e_std)
    # For pytorch > = 1.6.0
    bins = torch.linspace(hp.e_min, hp.e_max, steps=255).to(torch.device("cuda" if hp.ngpu > 0 else "cpu"))

    e_quantize = torch.bucketize(e, bins)

    return F.one_hot(e_quantize.long(), 256).float()