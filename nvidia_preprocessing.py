import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from utils.stft import TacotronSTFT
from utils.util import read_wav_np
from dataset.audio.audio_processing import pitch
from dataset.audio.pitch import Dio
from dataset.audio.energy import Energy
from utils.hparams import HParam


def main(args, hp):
    stft = TacotronSTFT(
        filter_length=hp.audio.n_fft,
        hop_length=hp.audio.hop_length,
        win_length=hp.audio.win_length,
        n_mel_channels=hp.audio.n_mels,
        sampling_rate=hp.audio.sample_rate,
        mel_fmin=hp.audio.fmin,
        mel_fmax=hp.audio.fmax,
    )

    wav_files = glob.glob(os.path.join(args.data_path, "**", "*.wav"), recursive=True)
    mel_path = os.path.join(hp.data.data_dir, "mels")
    energy_path = os.path.join(hp.data.data_dir, "energy")
    pitch_path = os.path.join(hp.data.data_dir, "pitch")
    os.makedirs(mel_path, exist_ok=True)
    os.makedirs(energy_path, exist_ok=True)
    os.makedirs(pitch_path, exist_ok=True)
    print("Sample Rate : ", hp.audio.sample_rate)
    for wavpath in tqdm.tqdm(wav_files, desc="preprocess wav to mel"):
        sr, wav = read_wav_np(wavpath, hp.audio.sample_rate)
        p = pitch(wav, hp)  # [T, ] T = Number of frames
        wav = torch.from_numpy(wav).unsqueeze(0)
        mel, mag = stft.mel_spectrogram(wav)  # mel [1, 80, T]  mag [1, num_mag, T]
        mel = mel.squeeze(0)  # [num_mel, T]
        mag = mag.squeeze(0)  # [num_mag, T]
        e = torch.norm(mag, dim=0)  # [T, ]
        p = p[: mel.shape[1]]
        id = os.path.basename(wavpath).split(".")[0]
        np.save("{}/{}.npy".format(mel_path, id), mel.numpy(), allow_pickle=False)
        np.save("{}/{}.npy".format(energy_path, id), e.numpy(), allow_pickle=False)
        np.save("{}/{}.npy".format(pitch_path, id), p, allow_pickle=False)


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
