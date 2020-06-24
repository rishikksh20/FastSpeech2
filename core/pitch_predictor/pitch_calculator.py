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
    # f0_numpy = f0.cpu().detach().numpy()
    # if is_training:
    #     f0_numpy[f0_numpy < 1] = 1
    #     # bins = np.logspace(0, np.log10(f0.max()), 256)
    #     log_f0 = np.log(f0_numpy)
    # else:
    #     log_f0 = f0_numpy
    # bins = np.linspace(np.log(hp.p_min), np.log(hp.p_max), num=256)
    #
    # p_quantize = np.digitize(log_f0, bins)
    # p_quantize = torch.from_numpy(p_quantize -1 ).float().to(torch.device("cuda" if hp.ngpu > 0 else "cpu"))

    # Required pytorch >= 1.6.0
    if is_training:
        f0 = f0 + 1 # convert 0 to 1 because log 0 == nan
        # bins = np.logspace(0, np.log10(f0.max()), 256)
        log_f0 = torch.log(f0)
    else:
        log_f0 = f0

    bins = torch.linspace(torch.log(hp.p_min), torch.log(hp.p_max+1), steps=256).to(torch.device("cuda" if hp.ngpu > 0 else "cpu"))
    p_quantize = torch.bucketize(log_f0, bins)
    p_quantize = p_quantize - 1  # -1 to convert 1 to 256 --> 0 to 255
    return F.one_hot(p_quantize.long(), 256).float()

