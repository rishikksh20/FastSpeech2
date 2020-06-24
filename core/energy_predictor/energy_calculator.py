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

    # bins = np.linspace(hp.e_min, hp.e_max, num=256)
    # e_numpy = e.cpu().detach().numpy()
    # e_quantize = np.digitize(e_numpy, bins)
    # e_quantize = torch.from_numpy(e_quantize-1).to(torch.device("cuda" if hp.ngpu > 0 else "cpu")) # -1 to convert 1 to 256 --> 0 to 255

    # For pytorch > = 1.6.0
    bins = torch.linspace(hp.e_min, hp.e_max, steps=256).to(torch.device("cuda" if hp.ngpu > 0 else "cpu"))
    e_quantize = torch.bucketize(e, bins)
    e_quantize = e_quantize -1 # -1 to convert 1 to 256 --> 0 to 255
    return F.one_hot(e_quantize.long(), 256).float()