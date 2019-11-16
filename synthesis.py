"""TTS Inference script."""

import configargparse
import logging
import os
import torch
import json
import sys
from utils.util import set_deterministic_pytorch
from transformer import Transformer
import hparams as hp
from dataset.texts import text_to_sequence
import numpy as np
import time
import argparse
from dataset.audio_processing import reconstruct_waveform
from dataset.audio_processing import save_wav
from train import num_params

def synthesis(args):
    """Decode with E2E-TTS model."""
    set_deterministic_pytorch(args)
    # read training config
    idim = hp.symbol_len
    odim = hp.num_mels
    model = Transformer(idim, odim, args)
    num_params(model)
    print(model)
    # load trained model parameters
    #logging.info('reading model parameters from ' + args.model)
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

    input = np.asarray(text_to_sequence(args.text.strip(), hp.tts_cleaner_names))
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
        print("probs size : ", probs.size())
        print("attn size : ", att_ws.size())

        logging.info("inference speed = %s msec / frame." % (
            (time.time() - start_time) / (int(outs.size(0)) * 1000)))
        if outs.size(0) == text.size(0) * args.maxlenratio:
            logging.warning("output length reaches maximum length .")
            
        print("mels",outs.size())
        mel = outs.cpu().numpy() # [T_out, num_mel]
        print("numpy ",mel.shape)
        # plot prob and att_ws
        if probs is not None:
            print("plot probs")
            _plot_and_save(probs.cpu().numpy(),"results/probs/{}_prob.png".format(idx))
        if att_ws is not None:
            _plot_and_save(att_ws.cpu().numpy(), "results/att_ws/{}_att_ws.png".format(idx))

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
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    # task related
    parser.add_argument('--text', type=str, required=True,
                        help='Filename of train label data (json)')
    parser.add_argument('--path', type=str, required=True,
                        help='Model file parameters to read')
    parser.add_argument('--model-conf', type=str, default=None,
                        help='Model config file')
    # decoding related
    parser.add_argument('--maxlenratio', type=float, default=5,
                        help='Maximum length ratio in decoding')
    parser.add_argument('--minlenratio', type=float, default=0,
                        help='Minimum length ratio in decoding')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold value in decoding')
    return parser


def main(args):
    """Run deocding."""
    parser = get_parser()
    args = parser.parse_args(args)


    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    audio = synthesis(args)
    m = audio.T
    m = (m + 4) / 8
    np.clip(m, 0, 1, out=m)
    wav = reconstruct_waveform(m, n_iter=60)
    save_path = '{}/demo_200_v1k.wav'.format(args.out)
    save_wav(wav, save_path)


def get_model_conf(model_path, conf_path=None):
    """Get model config information by reading a model config file (model.json).
    Args:
        model_path (str): Model path.
        conf_path (str): Optional model config path.
    Returns:
        list[int, int, dict[str, Any]]: Config information loaded from json file.
    """
    if conf_path is None:
        model_conf = os.path.dirname(model_path) + '/model.json'
    else:
        model_conf = conf_path
    with open(model_conf, "rb") as f:
        logging.info('reading a config file from ' + model_conf)
        confs = json.load(f)
    if isinstance(confs, dict):
        # for lm
        args = confs
        return argparse.Namespace(**args)
    else:
        # for asr, tts, mt
        idim, odim, args = confs
        return idim, odim, argparse.Namespace(**args)

if __name__ == '__main__':
    print("Starting")
    main(sys.argv[1:])
