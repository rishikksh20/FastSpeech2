import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def save_attention(attn, path):
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(attn.T, interpolation="nearest", aspect="auto")
    fig.savefig(f"{path}.png", bbox_inches="tight")
    plt.close(fig)


def save_spectrogram(M, path, length=None):
    M = np.flip(M, axis=0)
    if length:
        M = M[:, :length]
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(M, interpolation="nearest", aspect="auto")
    fig.savefig(f"{path}.png", bbox_inches="tight")
    plt.close(fig)


def plot(array):
    fig = plt.figure(figsize=(30, 5))
    ax = fig.add_subplot(111)
    ax.xaxis.label.set_color("grey")
    ax.yaxis.label.set_color("grey")
    ax.xaxis.label.set_fontsize(23)
    ax.yaxis.label.set_fontsize(23)
    ax.tick_params(axis="x", colors="grey", labelsize=23)
    ax.tick_params(axis="y", colors="grey", labelsize=23)
    plt.plot(array)


def plot_spec(M):
    M = np.flip(M, axis=0)
    plt.figure(figsize=(18, 4))
    plt.imshow(M, interpolation="nearest", aspect="auto")
    plt.show()


def plot_image(target, melspec, mel_lengths):  # , alignments
    fig, axes = plt.subplots(2, 1, figsize=(20, 20))
    T = mel_lengths[-1]

    axes[0].imshow(target[-1].T.detach().cpu()[:, :T], origin="lower", aspect="auto")

    axes[1].imshow(melspec.cpu()[:, :T], origin="lower", aspect="auto")

    return fig


def save_figure_to_numpy(fig, spectrogram=False):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if spectrogram:
        return data
    data = np.transpose(data, (2, 0, 1))
    return data


def plot_waveform_to_numpy(waveform):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot()
    ax.plot(range(len(waveform)), waveform, linewidth=0.1, alpha=0.7, color="blue")

    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.ylim(-1, 1)
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig, True)
    plt.close()
    return data


def generate_audio(mel, vocoder):
    # input mel shape - [1,80,T]
    vocoder.eval()
    if torch.cuda.is_available():
        vocoder = vocoder.cuda()
        mel = mel.cuda()

    with torch.no_grad():
        audio = vocoder.inference(mel)
    return audio
