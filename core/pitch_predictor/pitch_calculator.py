import torch
import hparams as hp
import pyworld as pw
import numpy as np
import torch.nn.functional as F

def calculate_pitch(y):
    # Extract Pitch/f0 from raw waveform using PyWORLD
    y = y.astype(np.float64)
    f0, _ = pw.dio(y, 22050, frame_period=11.6)  # For hop size 256 frame period is 11.6 ms
    return pitch_to_one_hot(f0)

def retreive_pitch(file):

    f0= np.load(file)
    return pitch_to_one_hot(f0)

def pitch_to_one_hot(f0, is_training = True):
    # Required pytorch >= 1.6.0

    bins = torch.exp(torch.linspace(np.log(hp.p_min), np.log(hp.p_max), 255)).to(torch.device("cuda" if hp.ngpu > 0 else "cpu"))
    p_quantize = torch.bucketize(f0, bins)
    #p_quantize = p_quantize - 1  # -1 to convert 1 to 256 --> 0 to 255
    return F.one_hot(p_quantize.long(), 256).float()

