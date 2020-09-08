import configargparse
from utils.hparams import HParam
from dataset import dataloader as loader
from fastspeech import FeedForwardTransformer
import numpy as np
from utils.stft import TacotronSTFT
from dataset.audio_processing import pitch
from utils.hparams import HParam
import sys
import torch

"""def extract_feat(audio, hp):
    stft = TacotronSTFT(
        filter_length=hp.audio.n_fft,
        hop_length=hp.audio.hop_length,
        win_length=hp.audio.win_length,
        n_mel_channels=hp.audio.n_mels,
        sampling_rate=hp.audio.sample_rate,
        mel_fmin=hp.audio.fmin,
        mel_fmax=hp.audio.fmax,
    )
    p = pitch(wav, hp)
    wav = torch.from_numpy(wav).unsqueeze(0)
    mel, mag = stft.mel_spectrogram(wav)  # mel [1, 80, T]  mag [1, num_mag, T]
    mel = mel.squeeze(0)  # [num_mel, T]
    mag = mag.squeeze(0)  # [num_mag, T]
    e = torch.norm(mag, dim=0)  # [T, ]
    p = p[: mel.shape[1]]

    return p, e
    """

def evaluate(args, hp):

    energy_diff = list()
    pitch_diff = list()

    validloader = loader.get_tts_dataset(hp.data.data_dir, 1, hp, True)
    print("Checkpoint : ", args.checkpoint_path)

    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
    else:
        logging.info("Checkpoint not exixts")
        return None

    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint["hp_str"])

    idim = len(valid_symbols)
    odim = hp.audio.num_mels
    model = FeedForwardTransformer(
        idim, odim, hp
    )
    os.makedirs(args.out, exist_ok=True)
    if args.old_model:
        logging.info("\nSynthesis Session...\n")
        model.load_state_dict(checkpoint, strict=False)
    else:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
    for valid in validloader:
        x_, input_length_, y_, _, out_length_, ids_, dur_, e_, p_ = valid
        model.eval()
        with torch.no_grad():
            loss_, report_dict_, _ = model(
                x_.cuda(),
                input_length_.cuda(),
                y_.cuda(),
                out_length_.cuda(),
                dur_.cuda(),
                e_.cuda(),
                p_.cuda(),
            )
            ilens = torch.tensor([x_[-1].shape[0]], dtype=torch.long, device=x.device)
            xs = x_[-1].unsqueeze(0)
            _, after_outs, d_outs, e_outs, p_outs = self._forward(xs, x_[-1].cuda())  # [T, num_mel]

            e_orig = model.energy_predictor.to_one_hot(e_)
            p_orig = model.pitch_predictor.to_one_hot(p_)
            energy_diff.append(e_orig - e_outs)
            pitch_diff.append(p_orig - p_outs)

        audio = generate_audio(
            mels_.unsqueeze(0), vocoder
        )  # selecting the last data point to match mel generated above
        audio = audio.cpu().float().numpy()
        audio = audio / (
            audio.max() - audio.min()
        )  # get values between -1 and 1

        '''_, target = read_wav_np( hp.data.wav_dir + f"{ids_[-1]}.wav", sample_rate=hp.audio.sample_rate)
        target_pitch = np.load(hp.data.data_dir + f"pitch/{ids_[-1]}.wav" )
        target_energy = np.load(hp.data.data_dir + f"energy/{ids_[-1]}.wav" )
        '''

    np.save(args.outdir + "score_pitch", target_pitch.numpy())
    np.save(args.outdir + "score_energy", target_energy.numpy())
    return energy_diff, pitch_diff


def get_parser():
    """Get parser of training arguments."""
    parser = configargparse.ArgumentParser(
        description="Train a new text-to-speech (TTS) model on one CPU, one or multiple GPUs",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )
    parser.add_argument(
        "-p",
        "--checkpoint_path",
        type=str,
        default=None,
        help="path of checkpoint pt to evaluate",
    )

    parser.add_argument("--outdir", type=str, required=True, help="Output directory")

    return parser

def main(cmd_args):
    """Run training."""
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)
    args = parser.parse_args(cmd_args)

    hp = HParam(args.config)
    with open(args.config, "r") as f:
        hp_str = "".join(f.readlines())

    evaluate(args, hp)
    

if __name__ == "__main__":
    main(sys.argv[1:])
