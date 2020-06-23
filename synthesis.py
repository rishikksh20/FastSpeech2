"""TTS Inference script."""

import configargparse
import logging
import os
import torch
import sys
from utils.util import set_deterministic_pytorch
from fastspeech import FeedForwardTransformer
import hparams as hp
from dataset.texts import  phonemes_to_sequence
import time
from dataset.audio_processing import reconstruct_waveform, griffin_lim
from dataset.audio_processing import save_wav
import librosa
import numpy as np
from utils.stft import STFT

def synthesis(args, text):
    """Decode with E2E-TTS model."""
    set_deterministic_pytorch(args)
    # read training config
    idim = hp.symbol_len
    odim = hp.num_mels
    model = FeedForwardTransformer(idim, odim, args)
    print(model)

    if os.path.exists(args.path):
        print('\nSynthesis Session...\n')
        model.load_state_dict(torch.load(args.path), strict=False)
    else:
        print("Checkpoint not exixts")
        return None

    model.eval()

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    input = np.asarray(phonemes_to_sequence(text.split()))
    text = torch.LongTensor(input)
    text = text.cuda()
    #[num_char]

        # define function for plot prob and att_ws
    def _plot_and_save(array, figname, figsize=(6, 4), dpi=150):
        import matplotlib.pyplot as plt
        shape = array.shape
        if len(shape) == 1:
            # for eos probability
            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(array)
            plt.xlabel("Frame")
            plt.ylabel("Probability")
            plt.ylim([0, 1])
        elif len(shape) == 2:
            # for tacotron 2 attention weights, whose shape is (out_length, in_length)
            plt.figure(figsize=figsize, dpi=dpi)
            plt.imshow(array, aspect="auto")
            plt.xlabel("Input")
            plt.ylabel("Output")
        elif len(shape) == 4:
            # for transformer attention weights, whose shape is (#leyers, #heads, out_length, in_length)
            plt.figure(figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi)
            for idx1, xs in enumerate(array):
                for idx2, x in enumerate(xs, 1):
                    plt.subplot(shape[0], shape[1], idx1 * shape[1] + idx2)
                    plt.imshow(x, aspect="auto")
                    plt.xlabel("Input")
                    plt.ylabel("Output")
        else:
            raise NotImplementedError("Support only from 1D to 4D array.")
        plt.tight_layout()
        if not os.path.exists(os.path.dirname(figname)):
            # NOTE: exist_ok = True is needed for parallel process decoding
            os.makedirs(os.path.dirname(figname), exist_ok=True)
        plt.savefig(figname)
        plt.close()

    with torch.no_grad():
        # decode and write
        idx = input[:5]
        start_time = time.time()
        print("text :", text.size())
        outs, probs, att_ws = model.inference(text, args)
        print("Out size : ",outs.size())

        logging.info("inference speed = %s msec / frame." % (
            (time.time() - start_time) / (int(outs.size(0)) * 1000)))
        if outs.size(0) == text.size(0) * args.maxlenratio:
            logging.warning("output length reaches maximum length .")
            
        print("mels",outs.size())
        mel = outs.cpu().numpy() # [T_out, num_mel]
        print("numpy ",mel.shape)
        

        return mel






# NOTE: you need this func to generate our sphinx doc
def get_parser():
    """Get parser of decoding arguments."""
    parser = configargparse.ArgumentParser(
        description='Synthesize speech from text using a TTS model on one CPU',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    # general configuration

    parser.add_argument('--ngpu', default=1, type=int,
                        help='Number of GPUs')
    parser.add_argument('--backend', default='pytorch', type=str,
                        choices=['chainer', 'pytorch'],
                        help='Backend library')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--out', type=str, required=True,
                        help='Output filename')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    # task related
    parser.add_argument('--text', type=str, required=True,
                        help='Filename of train label data (json)')
    parser.add_argument('--path', type=str, required=True,
                        help='Model file parameters to read')
    return parser


EPS = 1e-10


def logmelspc_to_linearspc(lmspc, fs, n_mels, n_fft, fmin=None, fmax=None):
    """Convert log Mel filterbank to linear spectrogram.

    Args:
        lmspc (ndarray): Log Mel filterbank (T, n_mels).
        fs (int): Sampling frequency.
        n_mels (int): Number of mel basis.
        n_fft (int): Number of FFT points.
        f_min (int, optional): Minimum frequency to analyze.
        f_max (int, optional): Maximum frequency to analyze.

    Returns:
        ndarray: Linear spectrogram (T, n_fft // 2 + 1).

    """
    assert lmspc.shape[1] == n_mels
    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax
    mspc = np.power(10.0, lmspc)
    mel_basis = librosa.filters.mel(fs, n_fft, n_mels, fmin, fmax)
    inv_mel_basis = np.linalg.pinv(mel_basis)
    spc = np.maximum(EPS, np.dot(inv_mel_basis, mspc.T).T)

    return spc


def griffin_lim_(spc, n_fft, n_shift, win_length, window='hann', n_iters=100):
    """Convert linear spectrogram into waveform using Griffin-Lim.

    Args:
        spc (ndarray): Linear spectrogram (T, n_fft // 2 + 1).
        n_fft (int): Number of FFT points.
        n_shift (int): Shift size in points.
        win_length (int): Window length in points.
        window (str, optional): Window function type.
        n_iters (int, optionl): Number of iterations of Griffin-Lim Algorithm.

    Returns:
        ndarray: Reconstructed waveform (N,).

    """
    # assert the size of input linear spectrogram
    assert spc.shape[1] == n_fft // 2 + 1
    spc = np.abs(spc.T)
    y = librosa.griffinlim(
        S=spc,
        n_iter=n_iters,
        hop_length=n_shift,
        win_length=win_length,
        window=window
    )
    return y



# For TTS engine setup

def synthesis_tts(args, text, path):
    """Decode with E2E-TTS model."""
    set_deterministic_pytorch(args)
    print("TTS synthesis")
    # read training config
    idim = hp.symbol_len
    odim = hp.num_mels
    model = FeedForwardTransformer(idim, odim)

    if os.path.exists(path):
        logging.info('\nSynthesis Session...\n')
        model.load_state_dict(torch.load(path), strict=False)
    else:
        logging.info("Checkpoint not exixts")
        return None

    model.eval()

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)
    print("Text :",text)
    input = np.asarray(phonemes_to_sequence(text.split()))
    print("Input :",input)
    text = torch.LongTensor(input)
    text = text.cuda()
    #[num_char]

        # define function for plot prob and att_ws
    def _plot_and_save(array, figname, figsize=(6, 4), dpi=150):
        import matplotlib.pyplot as plt
        shape = array.shape
        if len(shape) == 1:
            # for eos probability
            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(array)
            plt.xlabel("Frame")
            plt.ylabel("Probability")
            plt.ylim([0, 1])
        elif len(shape) == 2:
            # for tacotron 2 attention weights, whose shape is (out_length, in_length)
            plt.figure(figsize=figsize, dpi=dpi)
            plt.imshow(array, aspect="auto")
            plt.xlabel("Input")
            plt.ylabel("Output")
        elif len(shape) == 4:
            # for transformer attention weights, whose shape is (#leyers, #heads, out_length, in_length)
            plt.figure(figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi)
            for idx1, xs in enumerate(array):
                for idx2, x in enumerate(xs, 1):
                    plt.subplot(shape[0], shape[1], idx1 * shape[1] + idx2)
                    plt.imshow(x, aspect="auto")
                    plt.xlabel("Input")
                    plt.ylabel("Output")
        else:
            raise NotImplementedError("Support only from 1D to 4D array.")
        plt.tight_layout()
        if not os.path.exists(os.path.dirname(figname)):
            # NOTE: exist_ok = True is needed for parallel process decoding
            os.makedirs(os.path.dirname(figname), exist_ok=True)
        plt.savefig(figname)
        plt.close()

    with torch.no_grad():
        # decode and write
        idx = input[:5]
        start_time = time.time()
        print("pridicting")
        outs, probs, att_ws = model.inference(text, args)

        logging.info("inference speed = %s msec / frame." % (
            (time.time() - start_time) / (int(outs.size(0)) * 1000)))
        if outs.size(0) == text.size(0) * 5:
            logging.warning("output length reaches maximum length .")

        mel = outs#.cpu().numpy() # [T_out, num_mel]

        

        return mel






# NOTE: you need this func to generate our sphinx doc
def get_parser_tts():
    """Get parser of decoding arguments."""
    parser = configargparse.ArgumentParser(
        description='Synthesize speech from text using a TTS model on one CPU',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    # general configuration

    parser.add_argument('--ngpu', default=1, type=int,
                        help='Number of GPUs')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--out', type=str,
                        help='Output filename')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    # task related
    parser.add_argument('--text', type=str,
                        help='Filename of train label data (json)')
    parser.add_argument('--path', type=str,
                        help='Model file parameters to read')
    return parser

def infer(text):
    args = sys.argv[1:]
    parser = get_parser_tts()
    args = parser.parse_args(args)

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))
    path = "./checkpoints/checkpoint_48k_steps.pyt"
    out = "results/"
    print("Text : ", text)
    if hp.melgan_vocoder:
        m = m.unsqueeze(0)
        vocoder = torch.hub.load('seungwonpark/melgan', 'melgan')
        vocoder.eval()
        if torch.cuda.is_available():
            vocoder = vocoder.cuda()
            mel = m.cuda()

        with torch.no_grad():
            wav = vocoder.inference(mel) # mel ---> batch, num_mels, frames [1, 80, 234]
            wav = wav.cpu().numpy()
    else:
        stft = STFT(filter_length=1024, hop_length=256, win_length=1024)
        print(m.size())
        m = m.unsqueeze(0)
        wav = griffin_lim(m, stft, 30)
        wav = wav.cpu().numpy()
    save_path = '{}/test_tts.wav'.format(out)
    save_wav(wav, save_path)
    return save_path

def main(args):
    """Run deocding."""
    parser = get_parser()
    args = parser.parse_args(args)
    stats_file = "checkpoints/stats.npy"

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    path = "./checkpoints/checkpoint_54k_steps.pyt"
    out = "results/"
    print("Text : ", args.text)
    audio = synthesis_tts(args, args.text, path)
    m = audio.T
    
    np.save("mel.npy", m.cpu().numpy())
    m = m.cpu().numpy()
    m = (m + 4) / 8
    wav = reconstruct_waveform(m, n_iter=60)
    
    save_path = '{}/test.wav'.format(args.out)
    save_wav(wav, save_path)

if __name__ == '__main__':
    print("Starting")
    main(sys.argv[1:])
