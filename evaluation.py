import configargparse
from dataset import dataloader as loader
from fastspeech import FeedForwardTransformer
import sys
import torch
from dataset.texts import valid_symbols
import os 
from utils.hparams import HParam, load_hparam_str
import numpy as np


def evaluate(hp, validloader, model):
    energy_diff = list()
    pitch_diff = list()
    dur_diff = list()

    l1 = torch.nn.L1Loss()
    model.eval()
    for valid in validloader:
        x_, input_length_, y_, _, out_length_, ids_, dur_, e_, p_, p_avg_, p_std_, p_cwt_cont_ = valid
        
        with torch.no_grad():
            ilens = torch.tensor([x_[-1].shape[0]], dtype=torch.long, device=x_.device)
            _, after_outs, d_outs, e_outs, p_outs, p_avg_outs, p_std_outs = model._forward(x_.cuda(), ilens.cuda(), out_length_.cuda(), dur_.cuda(), es=e_.cuda(), ps=p_.cuda(), is_inference=False)  # [T, num_mel]

            # e_orig = model.energy_predictor.to_one_hot(e_).squeeze()
            # p_orig = model.pitch_predictor.to_one_hot(p_).squeeze()
            
            #print(d_outs)

            dur_diff.append(l1(d_outs, dur_.cuda()).item())      #.numpy()
            energy_diff.append(l1(e_outs, e_.cuda()).item())      #.numpy()
            pitch_diff.append(l1(p_outs, p_cwt_cont_.cuda()).item())       #.numpy()

            
        '''_, target = read_wav_np( hp.data.wav_dir + f"{ids_[-1]}.wav", sample_rate=hp.audio.sample_rate)
        target_pitch = np.load(hp.data.data_dir + f"pitch/{ids_[-1]}.wav" )
        target_energy = np.load(hp.data.data_dir + f"energy/{ids_[-1]}.wav" )
        '''
    model.train()
    return np.mean(pitch_diff), np.mean(energy_diff), np.mean(dur_diff)


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

    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
    else:
        print("Checkpoint not exixts")
        return None

    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint["hp_str"])
    
    validloader = loader.get_tts_dataset(hp.data.data_dir, 1, hp, True)
    print("Checkpoint : ", args.checkpoint_path)

    

    idim = len(valid_symbols)
    odim = hp.audio.num_mels
    model = FeedForwardTransformer(
        idim, odim, hp
    )
    # os.makedirs(args.out, exist_ok=True)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["model"])

    evaluate(hp, validloader, model)
    

if __name__ == "__main__":
    main(sys.argv[1:])
