import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from utils.stft import TacotronSTFT
from utils.util import read_wav_np
from utils.util import str_to_int_list
from dataset.audio.pitch import Dio
from dataset.audio.energy import Energy
from utils.hparams import HParam

def preprocess(args, hp, file):
    stft = TacotronSTFT(
        filter_length=hp.audio.n_fft,
        hop_length=hp.audio.hop_length,
        win_length=hp.audio.win_length,
        n_mel_channels=hp.audio.n_mels,
        sampling_rate=hp.audio.sample_rate,
        mel_fmin=hp.audio.fmin,
        mel_fmax=hp.audio.fmax,
    )

    energy = Energy()

    pitch = Dio()
    path = args.data_path
    with open("{}".format(hp.data.train_filelist), encoding="utf-8") as f:
        _metadata = [line.strip().split("|") for line in f]

    mel_path = os.path.join(hp.data.data_dir, "mels")
    energy_path = os.path.join(hp.data.data_dir, "energy")
    pitch_path = os.path.join(hp.data.data_dir, "pitch")

    os.makedirs(mel_path, exist_ok=True)
    os.makedirs(energy_path, exist_ok=True)
    os.makedirs(pitch_path, exist_ok=True)

    print("Sample Rate : ", hp.audio.sample_rate)

    for metadata in tqdm.tqdm(_metadata, desc="preprocess wav to mel"):
        wavpath = os.path.join(path, metadata[4])

        dur = str_to_int_list(metadata[2])
        dur = torch.from_numpy(np.array(dur))

        sr, wav = read_wav_np(wavpath, hp.audio.sample_rate)
        input = torch.from_numpy(wav)

        mel, mag = stft.mel_spectrogram(input.unsqueeze(0))  # mel [1, 80, T]  mag [1, num_mag, T]
        mel = mel.squeeze(0)  # [num_mel, T]
        mag = mag.squeeze(0)  # [num_mag, T]

        e = energy.forward(mag, dur)  # [T, ]
        p = pitch.forward(wav, mel.shape[1], dur)  # [T, ] T = Number of frames
        id = os.path.basename(wavpath).split(".")[0]

        np.save("{}/{}.npy".format(mel_path, id), mel.numpy(), allow_pickle=False)
        np.save("{}/{}.npy".format(energy_path, id), e.numpy(), allow_pickle=False)
        np.save("{}/{}.npy".format(pitch_path, id), p, allow_pickle=False)


def main(args, hp):
    print("Preprocess Training dataset :")
    preprocess(args, hp, hp.data.train_filelist)
    print("Preprocess Validation dataset :")
    preprocess(args, hp, hp.data.valid_filelist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path", type=str, required=True, help="root directory of wav files"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )
    args = parser.parse_args()

    hp = HParam(args.config)

    main(args, hp)
